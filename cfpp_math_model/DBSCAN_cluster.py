import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point, Polygon
import geopandas as gpd
import math

# 获取两个文件中的经纬度点坐标
def get_location(filename):

    cfpp_data = pd.read_csv(filename)

    # 过滤,保留置信度大于0.30的子部件，是否根据子部件动态设置不同阈值过滤
    # print(cfpp_data)
    cfpp_data = cfpp_data[cfpp_data['Score'] >= 0.300]
    # print(cfpp_data)

    cfpp_location = cfpp_data[['Longitude','Latitude']].values
    cfpp_datas = cfpp_data[['Label','Score']].values

    return cfpp_datas, cfpp_location

# 聚类函数
def DBSCAN_cluster(all_loaction):

    # 设置 DBSCAN 参数
    # eps: 邻域的大小，表示一个点附近的点距离
    # min_samples: 形成一个簇的最小点数
    db = DBSCAN(eps=0.002, min_samples=2).fit(all_loaction)

    # 获取每个点的聚类标签
    labels = db.labels_
    print(labels)

    # 可视化结果
    # plt.scatter(all_loaction[:, 0], all_loaction[:, 1], c=labels, cmap='viridis')
    # plt.show()
    return labels

# 根据聚类的簇生成面，为避免生成最小外接矩形，添加缓冲区
def point2planes(all_location, labels):

    # 将每个聚类中的点构造为 MultiPoint 对象
    clusters = [all_location[labels == i] for i in np.unique(labels) if i != -1]  # 排除噪声点

    # print(clusters)

    rectangles = []
    buffer_size = 0.0002  # 设置缓冲区的大小

    for cluster in clusters:

        multipoint = MultiPoint(cluster)
        bounding_box = multipoint.minimum_rotated_rectangle
            
        # 对最小外接矩形增加缓冲，扩展矩形的范围
        rectangle = bounding_box.buffer(buffer_size)  # buffer_size 控制扩展的大小
        rectangles.append(rectangle)

    # 将生成的矩形转换为 GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=rectangles)

    # 将结果导出为 Shapefile
    gdf.to_file("./cfpp_math_model/result/cfpp_potential.shp")

    # 可视化结果
    for rect in rectangles:
        if rect.geom_type == 'Polygon':
            x, y = rect.exterior.xy  # 访问 Polygon 的外部边界
            plt.fill(x, y, alpha=0.5, color='orange')
    plt.scatter(all_location[:, 0], all_location[:, 1], c=labels, cmap='viridis')
    plt.show()

    return clusters

# 根据聚类结果确定潜在区域并或者子部件信息
def get_object(cfpp_datas, cfpp_location, labels):
    # 根据聚类后的labels划分潜在区域
    clusters = [cfpp_location[labels == i] for i in np.unique(labels) if i != -1]  # 排除噪声点
    data_clusters = []
    for cluster in clusters:
        # print(cluster)
        cfpp_dict = {'condensing_tower':[], 'chimney':[], 'power_plant_unit':[]}
        for cfpp_object in cluster:
            # index = cfpp_location.index(cfpp_object)
            # print(cfpp_object)
            index = np.where((cfpp_location == cfpp_object).all(axis=1))[0]
            # print(index[0])
            # print(cfpp_datas[index[0]])
            object_data = cfpp_datas[index[0]]
            if object_data[0] == 0 or object_data[0] == 1 :
                cfpp_dict['condensing_tower'].append(object_data[1])
            elif object_data[0] == 2 or object_data[0] == 3 :
                cfpp_dict['chimney'].append(object_data[1])
            else :
                cfpp_dict['power_plant_unit'].append(object_data[1])
        
        data_clusters.append(cfpp_dict)
    
    print(data_clusters)

    return clusters, data_clusters


# 动态调整权重（包括缩减抑制）
def adjust_weights(base_weights, component_counts, gamma):
    total_count = sum(component_counts.values())
    adjusted_weights = {}
    
    for k, count in component_counts.items():
        
        # 计算缩减因子
        if count > 0.8 * total_count:  # 触发抑制条件（数量远大于其他部件）
            reduction = gamma * (count / total_count)
        else:
            reduction = 0
        
        # 调整后的权重
        adjusted_weights[k] = base_weights[k] * (1 - reduction)
    
    # # 动态调整，保持权重比例
    # weight_sum = sum(adjusted_weights.values())
    # for k in adjusted_weights:
    #     adjusted_weights[k] /= weight_sum
    
    return adjusted_weights


# 定义得分计算函数
def calculate_score(components, base_weights, gamma):
    
    # 统计子部件数量
    component_counts = {k: len(v) for k, v in components.items()}
    
    # 动态调整权重（包括缩减抑制）
    weights = adjust_weights(base_weights, component_counts, gamma)
    
    # 初始化得分
    scores = {}
    total_score = 0
    
    for component, confidences in components.items():
        weight = weights.get(component, 0)  # 获取对应权重
        N = len(confidences)  # 子部件数量
        if N == 0:
            scores[component] = 0
            continue
        
        # 平均置信度
        avg_confidence = sum(confidences) / N
        
        # 得分计算公式
        if component == 'power_plant_unit':  # 发电机组
            score = weights[component] * N * avg_confidence
        else:  # 冷凝塔和烟囱
            score = weights[component] * math.sqrt(N) * avg_confidence
        
        scores[component] = score
        # 累加得分
        total_score += score
    
    return scores, total_score


def save_clusters_scores(clusters,total_score_list,scores_list):
    
    rectangles = []
    scores = []  # 存储总得分

    # 存储每个子部件得分字典
    subcomponent_scores_condensing_tower = []
    subcomponent_scores_chimney = []
    subcomponent_scores_power_plant_unit = []
    subcomponent_name_list = list(scores_list[0].keys())

    centers = []  # 存储外接矩形的中心点

    # csv输出文件
    outdata = []

    buffer_size = 0.0002  # 设置缓冲区的大小
    num =0

    for cluster in clusters:

        multipoint = MultiPoint(cluster)
        bounding_box = multipoint.minimum_rotated_rectangle

        # 过滤得分
        if total_score_list[num] < 0.600:
            num = num + 1
            continue
            
        # 对最小外接矩形增加缓冲，扩展矩形的范围
        rectangle = bounding_box.buffer(buffer_size)  # buffer_size 控制扩展的大小

        rectangles.append(rectangle)  # 添加几何对象

        # 添加子部件得分字典
        subcomponent_scores_condensing_tower.append(scores_list[num][subcomponent_name_list[0]])
        subcomponent_scores_chimney.append(scores_list[num][subcomponent_name_list[1]])
        subcomponent_scores_power_plant_unit.append(scores_list[num][subcomponent_name_list[2]])

        # 计算外接矩形的中心点
        center = rectangle.centroid
        centers.append((center.x, center.y))  # 存储中心点的坐标


        scores.append(total_score_list[num])  # 添加总得分

        outdata.append([center.x, center.y, scores_list[num][subcomponent_name_list[0]], scores_list[num][subcomponent_name_list[1]], scores_list[num][subcomponent_name_list[2]], total_score_list[num]])

        num = num + 1

    # 将生成的矩形转换为 GeoDataFrame
    # 创建 GeoDataFrame，并将几何对象和得分作为属性添加
    gdf = gpd.GeoDataFrame({
        'geometry': rectangles,
        subcomponent_name_list[0] : subcomponent_scores_condensing_tower,
        subcomponent_name_list[1] : subcomponent_scores_chimney,
        subcomponent_name_list[2] : subcomponent_scores_power_plant_unit,
        'total_score': scores
    })

    # 将结果导出为 Shapefile
    gdf.to_file("./cfpp_math_model/result/shandong_0.835/0.60/cfpp_potential_libra_0.60.shp")


    # 结果保存至csv文件
    # df = pd.DataFrame(outdata, columns=["Longitude", "Latitude", subcomponent_name_list[0], subcomponent_name_list[1], subcomponent_name_list[2], 'total_score'])
    # df.to_csv("./cfpp_math_model/cfpp_potential_libra.csv", index=False)


if __name__ == '__main__':

    cfpp_datas, cfpp_location = get_location('C:\\Users\\14398\\Desktop\\shandong_0.835.csv')
    labels = DBSCAN_cluster(cfpp_location)
    # clusters = point2planes(cfpp_location,labels)
    clusters, data_clusters = get_object(cfpp_datas, cfpp_location, labels)

    # 计算每一个潜在区域的得分
    # 子部件权重
    weights = {
        'power_plant_unit': 0.5,   # 发电机组权重
        'condensing_tower': 0.3,   # 冷凝塔权重
        'chimney': 0.2             # 烟囱权重
    }
    gamma = 0.5  # 缩减因子

    # 便利保存得分结果
    scores_list = []
    total_score_list = []

    for num in range(len(data_clusters)) :

        # 计算结果
        scores, total_score = calculate_score(data_clusters[num], weights, gamma)

        # 保存结果
        scores_list.append(scores)
        total_score_list.append(total_score)

        # 输出结果
        print(f"聚类簇ID: {num}")
        print(f"子部件得分: {scores}")
        print(f"总得分: {total_score:.4f}")
    
    save_clusters_scores(clusters,total_score_list,scores_list)

    # print(list(scores_list[0].keys()))
    # weights.




    

