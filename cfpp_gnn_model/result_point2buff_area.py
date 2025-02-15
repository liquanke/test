from shapely.geometry import MultiPoint, Point, Polygon
import geopandas as gpd
import numpy as np
import pandas as pd


# 根据聚类的簇生成面，为避免生成最小外接矩形，添加缓冲区
def point2planes(all_location):

    # print(clusters)

    rectangles = []
    buffer_size = 0.001  # 设置缓冲区的大小

    for location in all_location:
        point = Point(location[0], location[1])
            
        # 创建一个稍大的矩形，以点为中心，宽度和高度可以根据需要调整
        rectangle = point.buffer(buffer_size)  # buffer_size 控制矩形的大小
        rectangles.append(rectangle)

    # 将生成的矩形转换为 GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=rectangles)

    # 将结果导出为 Shapefile
    gdf.to_file("C:\\Users\\14398\\Desktop\\shandong_all\\shandong_all.shp")

if __name__ == '__main__':
    predict_cfpp = pd.read_csv('C:\\Users\\14398\\Desktop\\shandong_all.csv')
    # predict_cfpp = predict_cfpp[predict_cfpp['tag']==1]
    print(predict_cfpp)
    # predict_cfpp = predict_cfpp.head(1000)
    # print(predict_cfpp)
    all_locations = predict_cfpp[['lon', 'lat']].values
    point2planes(all_locations)
    # print(all_locations)