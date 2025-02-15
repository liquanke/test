import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

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
    # print(labels)

    # 可视化结果
    # plt.scatter(all_loaction[:, 0], all_loaction[:, 1], c=labels, cmap='viridis')
    # plt.show()
    return labels


# 根据聚类结果确定潜在区域并或者子部件信息
def get_object(cfpp_datas, cfpp_location, labels):
    # 根据聚类后的labels划分潜在区域
    clusters = [cfpp_location[labels == i] for i in np.unique(labels) if i != -1]  # 排除噪声点
    data_clusters = []
    for cluster in clusters:
        # print(cluster)
        cfpp_data = []
        # cfpp_dict = {'condensing_tower':[], 'chimney':[], 'power_plant_unit':[]}
        for cfpp_object in cluster:
            # index = cfpp_location.index(cfpp_object)
            # print(cfpp_object)
            index = np.where((cfpp_location == cfpp_object).all(axis=1))[0]
            # print(index[0])
            # print(cfpp_datas[index[0]])
            object_data = cfpp_datas[index[0]]
            if object_data[0] == 0 or object_data[0] == 1 :
                cfpp_data.append([1,0,0,object_data[1]])
                # cfpp_dict['condensing_tower'].append(object_data[1])
            elif object_data[0] == 2 or object_data[0] == 3 :
                # cfpp_dict['chimney'].append(object_data[1])
                cfpp_data.append([0,1,0,object_data[1]])
            else :
                # cfpp_dict['power_plant_unit'].append(object_data[1])
                cfpp_data.append([0,0,1,object_data[1]])
        
        data_clusters.append(cfpp_data)
    
    # print(data_clusters)

    return clusters, data_clusters

def get_node_dataset(cfpp_datas, cfpp_location, labels):
    # 正负样本聚类簇的ID
    cfpp_potential = pd.read_csv('./cfpp_gnn_model/cfpp_potential_libra.csv')
    postive_cfpp_index = cfpp_potential[cfpp_potential['total_score']>0.7].index.values
    negative_cfpp_index = cfpp_potential[cfpp_potential['total_score']<0.3].index.values

    # 筛选正负样本子部件
    filter_cfpp_datas = []
    filter_location = []
    filter_y = []

    for num in range(len(cfpp_datas)):
        label = labels[num]
        if label in postive_cfpp_index :
            filter_cfpp_datas.append(cfpp_datas[num])
            filter_location.append(cfpp_location[num])
            filter_y.append(1)
        elif label in negative_cfpp_index :
            filter_cfpp_datas.append(cfpp_datas[num])
            filter_location.append(cfpp_location[num])
            filter_y.append(0)
        else:
            continue

    # 生成图数据 先节点再边
    node_features = []
    edge_index = []
    edge_attr = []

    print(0)
    distance_matrix = cdist(filter_location, filter_location, metric='euclidean')
    print(1)
    # 生成节点特征
    for num in range(len(filter_cfpp_datas)):
        cls = filter_cfpp_datas[num][0]
        if (cls == 0 or cls == 1): node_features.append([1, 0, 0, filter_cfpp_datas[num][1]])
        elif (cls == 2 or cls == 3): node_features.append([0, 1, 0, filter_cfpp_datas[num][1]])
        else: node_features.append([0, 0, 1, filter_cfpp_datas[num][1]])
    
    # 生成边的特征
    for i in range(len(node_features)):
        for j in range(len(node_features)):
            if i != j and distance_matrix[i, j] < 0.002:
                edge_index.append([i, j])
                weight = 1 / (1 + distance_matrix[i, j])
                edge_attr.append(weight)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # 转置为 (2, num_edges)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor(filter_y, dtype=torch.long)

    # 转换为 PyTorch Geometric Data 格式
    x = torch.tensor(node_features, dtype=torch.float)
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    print(x.shape)
    print(edge_index.shape)
    print(edge_attr.shape)
    print(y.shape)

    return graph_data


if __name__ == '__main__':
    cfpp_datas, cfpp_location = get_location('./cfpp_gnn_model/shandong_0.835.csv')
    labels = DBSCAN_cluster(cfpp_location)
    clusters, data_clusters = get_object(cfpp_datas, cfpp_location, labels)
    get_node_dataset(cfpp_datas, cfpp_location, labels)




# def generate_graph_from_csv(csv_file, distance_threshold=0.003):
#     """
#     根据CSV文件生成图数据,用于图神经网络模型训练。
    
#     参数：
#     - csv_file: str, CSV文件路径
#     - distance_threshold: float, 距离阈值，控制边的连接

#     返回：
#     - graph_data: PyTorch Geometric Data 对象
#     """
#     # 1. 读取CSV文件
#     df = pd.read_csv(csv_file).fillna(0)
#     df_data_filter, labels = class_tranformer(df)
#     # print(df.dtypes)
#     coordinates = df_data_filter[['Longitude','Latitude']].values
#     confidences = df_data_filter['Score'].values
#     class_data = df_data_filter['Class'].values
#     # print(class_data)

    
#     # print(coordinates)
#     # 位置坐标归一化
#     np.array(coordinates)
#     normalized_coordinates = (np.array(coordinates) - np.min(np.array(coordinates), axis=0)) / (np.max(np.array(coordinates), axis=0) - np.min(np.array(coordinates), axis=0))
#     # print(normalized_coordinates)

#     # 2. 编码子部件标签
#     encoder = OneHotEncoder(sparse=False)
#     label_one_hot = encoder.fit_transform(labels.reshape(-1, 1))
#     # print(label_one_hot)

#     # 3. 生成节点特征
#     node_features = np.hstack([label_one_hot, normalized_coordinates, confidences.reshape(-1, 1)])
#     print(node_features[0])

#     # 4. 计算节点间距离矩阵
#     distance_matrix = cdist(coordinates, coordinates, metric='euclidean')

#     # 5. 生成边及其权重
#     edge_index = []
#     edge_attr = []
#     for i in range(len(node_features)):
#         for j in range(len(node_features)):
#             if i != j and distance_matrix[i, j] < distance_threshold:
#                 edge_index.append([i, j])
#                 weight = 1 / (1 + distance_matrix[i, j])
#                 edge_attr.append(weight)

#     edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # 转置为 (2, num_edges)
#     edge_attr = torch.tensor(edge_attr, dtype=torch.float)
#     y = torch.tensor(class_data, dtype=torch.long)
#     # print(edge_index.shape)
#     # print(edge_attr.shape)
#     # print(y.shape)

#     # 6. 转换为 PyTorch Geometric Data 格式
#     x = torch.tensor(node_features, dtype=torch.float)
#     graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

#     return graph_data

# if __name__ == '__main__':
#     # 示例：使用生成图数据函数
#     csv_file_path = "./cfpp_gnn_model/data/shandong25_0.835.csv"  # 文件路径
#     graph_data = generate_graph_from_csv(csv_file_path)

#     # 输出图数据的基本信息
#     print("图数据节点特征维度:", graph_data.x.size())
#     print("图数据边数:", graph_data.edge_index.size(1))
#     print("图数据边权重样本:", graph_data.edge_attr[:5])
