import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from osgeo import gdal
import os
import geopandas as gpd
import rasterio
from rasterio import features
import xarray as xr
import rioxarray as rio
import shapefile as sf


class get_contours:
	def __init__(self,
				 img_path=None,
				 out_path='.',
				 shp_ground=None,
				 corners=None):
		self.img_path = img_path
		self.out_path = out_path
		self.corners = corners
		self.raster = rio.open_rasterio(self.img_path)
		self.crop_tiff_path = None
		if shp_ground is not None:
			self.vector = gpd.read_file(shp_ground)
		else:
			self.vector = None

	def crop_area(self):
		''' Crop area by image coordinates '''
		if self.corners == None:
			self.min_y = -602602.673
			self.max_y = -734038.285
			self.min_x = -1689945.934
			self.max_x = -1521308.818
		else:
			self.min_x = self.corners[0]
			self.min_y = self.corners[1]
			self.max_x = self.corners[2]
			self.max_y = self.corners[3]

		self.cropped_raster = self.raster.sel(y=slice(self.min_y, self.max_y), x=slice(self.min_x, self.max_x))
		self.crop_tiff_path = '{}/cropped_{}'.format(self.out_path, os.path.basename(self.img_path))
		self.cropped_raster.rio.to_raster(self.crop_tiff_path)

	def create_mask(self):
		''' Mask land area from a vector file '''

		# Extract a polygon
		sea_id = 'Grounded'
		sea_name = self.vector['NAME'].loc[self.vector.NAME == 'Grounded'].to_list()[0]
		out_raster = '{}/mask_{}.tiff'.format(self.out_path, os.path.basename(self.img_path))

		# Open example raster
		raster = rasterio.open(self.crop_tiff_path)

		# Reproject vector
		self.vector = self.vector.to_crs(raster.crs)

		# Get list of geometries for all features in vector file
		# geom = [shapes for shapes in vector.geometry]

		# Rasterize vector using the shape and coordinate system of the raster
		self.rasterized = features.rasterize(self.vector['geometry'].loc[self.vector.NAME == sea_id].to_list(),
											 out_shape=raster.shape,
											 fill=0,
											 out=None,
											 transform=raster.transform,
											 all_touched=False,
											 default_value=1,
											 dtype=None)
		# Write generated mask
		with rasterio.open(
				out_raster, "w",
				driver="GTiff",
				crs=raster.crs,
				transform=raster.transform,
				dtype=rasterio.uint8,
				count=1,
				width=raster.width,
				height=raster.height) as dst:
			dst.write(self.rasterized, indexes=1)

	def get_pixel_coordinates(self):
		# Get coordinates of all pixel in satellite image (cropped!)
		if self.crop_tiff_path is not None:
			with rasterio.open(self.crop_tiff_path) as src:
				band1 = src.read(1)
				print('Band1 has shape', band1.shape)
				height = band1.shape[0]
				width = band1.shape[1]
				cols, rows = np.meshgrid(np.arange(width), np.arange(height))
				xs, ys = rasterio.transform.xy(src.transform, rows, cols)
				self.lons = np.array(xs)
				self.lats = np.array(ys)
		else:
			with rasterio.open(self.raster) as src:
				band1 = src.read(1)
				print('Band1 has shape', band1.shape)
				height = band1.shape[0]
				width = band1.shape[1]
				cols, rows = np.meshgrid(np.arange(width), np.arange(height))
				xs, ys = rasterio.transform.xy(src.transform, rows, cols)
				self.lons = np.array(xs)
				self.lats = np.array(ys)

	def get_contour_coordinates(self):
		''' Get contour coordinates '''
		self.get_pixel_coordinates()
		self.latlon_contours = self.contours.copy()

		for i, contour in enumerate(self.contours):
			latlon_contour = np.zeros_like(self.contours[i])
			for j in range(latlon_contour.shape[0]):
				if contour[j][0] >= 0 and contour[j][1] >= 0:
					self.latlon_contours[i][j] = [self.lons[round(contour[j][0]), round(contour[j][1])],
												  self.lats[round(contour[j][0]), round(contour[j][1])]]

	def export_shapefile(self):
		''' Create shapefile with polygons '''

		self.shp_path = '{}/{}.shp'.format(self.out_path, os.path.basename(self.img_path))
		w = sf.Writer(self.shp_path, sf.POLYGON)
		w.field('id', 'C', '40')

		# Ex.:   w.poly([
		#        [[113,24], [112,32], [117,36], [122,37], [118,20]], # poly 1
		#        [[116,29],[116,26],[119,29],[119,32]], # hole 1
		#        [[15,2], [17,6], [22,7]]  # poly 2
		#        ])

		for i, contour in enumerate(self.latlon_contours):
			w.poly([contour.tolist()])
			w.record(str(i))

		# Save projection information
		tiff = rio.open_rasterio(self.img_path)
		wkt_str = tiff.rio.crs.to_wkt()

		prj = open(f'{self.shp_path[:-4]}.prj', 'w')
		prj.write(wkt_str)
		prj.close()

		w.close()

	def get_contours(self, val=70):
		''' Find contours around icebergs and ice shelf front '''

		self.r = self.cropped_raster.values[0, :, :].astype('float')

		# Mask vector data
		self.r[self.rasterized == 1] = 0
		self.r[self.r == 0] = np.nan

		# Normalize to 0-255
		if np.nanmax(self.r) > 255:
			self.r = (self.r - np.nanmin(self.r)) * (255 / (np.nanmax(self.r) - np.nanmin(self.r)))
		else:
			pass

		# Find contours at a constant value
		self.contours = measure.find_contours(self.r, val)

		# 4. Display the image and plot all contours found
		fig, ax = plt.subplots()
		ax.imshow(self.r, cmap=plt.cm.gray)

		for contour in self.contours:
			ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

		ax.axis('image')
		ax.set_xticks([])
		ax.set_yticks([])
		plt.show()