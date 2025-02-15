from osgeo import gdal

path = "C:\\Users\\14398\\Desktop\\data\\gep\\output.tiff"
path1 = "C:\\Users\\14398\\Desktop\\ctpp_86_zoom_16.tif"
ds = gdal.Open(path)
geo_info = ds.GetGeoTransform()
print(geo_info)

coord_info = ds.GetProjection()
print(coord_info)