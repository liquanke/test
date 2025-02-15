import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point, Polygon
import geopandas as gpd

# 获取两个文件中的经纬度点坐标
def get_location():

    wri_data = pd.read_csv("china_coal_tpp.csv")
    gem_data = pd.read_csv("GEM_delete_repeat_china.csv")

    wri_location = wri_data[['longitude', 'latitude']].values
    gem_location = gem_data[['Longitude','Latitude']].values

    all_location = np.concatenate((np.array(wri_location), np.array(gem_location)), axis=0)

    print(all_location.shape)

    return all_location

# 聚类函数
def DBSCAN_cluster(all_loaction):

    # 设置 DBSCAN 参数
    # eps: 邻域的大小，表示一个点附近的点距离
    # min_samples: 形成一个簇的最小点数
    db = DBSCAN(eps=0.1, min_samples=1).fit(all_loaction)

    # 获取每个点的聚类标签
    labels = db.labels_
    print(labels)

    plt.scatter(all_loaction[:, 0], all_loaction[:, 1], c=labels, cmap='viridis')
    plt.show()
    return labels

# 根据聚类的簇生成面，为避免生成最小外接矩形，添加缓冲区
def point2planes(all_location, labels):

    # 将每个聚类中的点构造为 MultiPoint 对象
    clusters = [all_location[labels == i] for i in np.unique(labels) if i != -1]  # 排除噪声点

    # print(clusters)

    rectangles = []
    buffer_size = 0.03  # 设置缓冲区的大小
    buffer_size1 = 0.02  # 设置缓冲区的大小

    for cluster in clusters:
        if len(cluster) == 1:  # 如果聚类中只有一个点
            point = Point(cluster[0][0], cluster[0][1])
            
            # 创建一个稍大的矩形，以点为中心，宽度和高度可以根据需要调整
            rectangle = point.buffer(buffer_size)  # buffer_size 控制矩形的大小
            rectangles.append(rectangle)
        else:
            multipoint = MultiPoint(cluster)
            bounding_box = multipoint.minimum_rotated_rectangle
            
            # 对最小外接矩形增加缓冲，扩展矩形的范围
            rectangle = bounding_box.buffer(buffer_size1)  # buffer_size 控制扩展的大小
            rectangles.append(rectangle)

    # 将生成的矩形转换为 GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=rectangles)

    # 将结果导出为 Shapefile
    gdf.to_file("expanded_bounding_rectangles.shp")

    # 可视化结果
    for rect in rectangles:
        if rect.geom_type == 'Polygon':
            x, y = rect.exterior.xy  # 访问 Polygon 的外部边界
            plt.fill(x, y, alpha=0.5, color='orange')
    plt.scatter(all_location[:, 0], all_location[:, 1], c=labels, cmap='viridis')
    plt.show()

# 给生成的矩形框缓冲区添加省份字段
def assign_province():

    # 1. 加载省份和矩形数据
    provinces = gpd.read_file('./province/province.shp', encoding='utf-8')
    rectangles = gpd.read_file('expanded_bounding_rectangles.shp', encoding='utf-8')

    # # 2. 确保两个数据集的 CRS 一致
    # rectangles = rectangles.to_crs(provinces.crs)

    # 3. 计算每个矩形与每个省的重叠面积
    overlap_areas = []

    # 保存至csv的文件
    fid = 0
    csv_data = []

    for _, rect in rectangles.iterrows():
        rect_geom = rect['geometry']
        
        # 计算矩形与每个省的交集，并获取交集的面积
        max_area = 0
        assigned_province = None
        
        for _, province in provinces.iterrows():
            province_geom = province['geometry']
            
            # 计算矩形与省份的交集面积
            intersection = rect_geom.intersection(province_geom)
            intersection_area = intersection.area
            
            # 如果交集面积更大，则更新最大面积及对应的省份
            if intersection_area > max_area:
                max_area = intersection_area
                print(province['pr_name'])
                assigned_province = province['pr_name']  # 假设省份字段名为 'province_name'
        
        # 将结果存储到 overlap_areas 中
        overlap_areas.append({'geometry': rect_geom, 'province': assigned_province})

        csv_data.append(assigned_province)
        fid = fid + 1

    # 4. 将结果转换为 GeoDataFrame
    assigned_rectangles = gpd.GeoDataFrame(overlap_areas, geometry='geometry')

    # 5. 导出为 Shapefile
    assigned_rectangles.to_file('./assign_province/rectangles_assigned_to_provinces.shp', encoding='utf-8')

    # 导出为csv文件与shp文件对应
    csv_pd = pd.DataFrame(csv_data, columns= ['province'])
    csv_pd.to_csv('fid_province.csv', encoding='utf_8_sig')

    # 6. 可视化结果
    ax = provinces.plot(color='lightgray', figsize=(10, 10))
    assigned_rectangles.plot(ax=ax, color='orange', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    # all_location = get_location()
    # labels = DBSCAN_cluster(all_location)
    # point2planes(all_location, labels)
    assign_province()