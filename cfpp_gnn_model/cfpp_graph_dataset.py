import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point, Polygon
import geopandas as gpd
import math
from scipy.spatial.distance import cdist
import torch
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

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

def generate_graph_dataset(clusters, data_clusters):

    # 生成图数据（batch）
    all_x = []
    all_adj = []
    for num in range(len(clusters)):
        x = data_clusters[num]
        distance_matrix = cdist(clusters[num], clusters[num], metric='euclidean')
        edge_index = []
        edge_attr = []
        # print(num)
        for i in range(len(clusters[num])):
            for j in range(len(clusters[num])):
                if i != j and distance_matrix[i, j] < 0.002:
                    edge_index.append([i, j])
                    weight = 1 / (1 + distance_matrix[i, j])
                    edge_attr.append(weight)
                    # print(i,j)
        # 邻接矩阵转换
        adj = np.zeros((len(clusters[num]), len(clusters[num])))
        for i in range(len(edge_index)):
            row, col = edge_index[i]
            adj[row][col] = 1
        
        # print(adj)
        # print(data_clusters[num])
        all_x.append(x)
        all_adj.append(adj)

    all_x = np.array(all_x)
    all_adj = np.array(all_adj)

    print(all_x.shape)
    print(all_adj.shape)

    # 对上述生成的图列表依据数学模型前期得到的潜力结果进行筛选，筛选两头分别作为正样本和负样本

    cfpp_potential = pd.read_csv('./cfpp_gnn_model/cfpp_potential_libra.csv')

    postive_cfpp_index = cfpp_potential[cfpp_potential['total_score']>0.7].index.values
    negative_cfpp_index = cfpp_potential[cfpp_potential['total_score']<0.3].index.values
    # print(type(postive_cfpp_index), postive_cfpp_index)
    # print(len(all_x))
    # print(cfpp_potential[cfpp_potential['total_score']<0.3])
    # 正样本
    postive_x = all_x[postive_cfpp_index]
    postive_adj = all_adj[postive_cfpp_index]
    postive_y = [1 for i in range(len(postive_cfpp_index))]

    # print(postive_x[0])
    # print(len(postive_x))
    # print(postive_y)
    # print(len(postive_y))

    # 负样本
    negative_x = all_x[negative_cfpp_index]
    negative_adj = all_adj[negative_cfpp_index]
    negative_y = [0 for i in range(len(negative_cfpp_index))]

    # 正负样本合并
    merge_x = np.append(postive_x, negative_x, axis=0)
    merge_adj = np.append(postive_adj, negative_adj, axis=0)
    merge_y = np.append(postive_y, negative_y, axis=0)

    # np.save('./cfpp_gnn_model/merge_x.npy',merge_x)
    # np.save('./cfpp_gnn_model/merge_adj.npy',merge_adj)
    # np.save('./cfpp_gnn_model/merge_y.npy',merge_y)


    # np.save('./cfpp_gnn_model/all_x.npy',all_x)
    # np.save('./cfpp_gnn_model/all_adj.npy',all_adj)



class GraphDataset(Dataset):
    def __init__(self, x_path, adj_path, y_path, transform=None):
        """
        初始化数据集
        :param x_path: 节点特征的路径 (.npy 文件)
        :param adj_path: 邻接矩阵的路径 (.npy 文件)
        :param y_path: 标签的路径 (.npy 文件)
        :param transform: 图数据的变换（可选）
        """
        self.node_features = np.load(x_path, allow_pickle=True)  # 加载节点特征
        self.adj_matrices = np.load(adj_path, allow_pickle=True)  # 加载邻接矩阵
        if y_path == None:
            length = len(self.node_features)
            self.labels = np.zeros(length)
        else:
            self.labels = np.load(y_path, allow_pickle=True)  # 加载标签
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        获取单个图的数据
        """
        if isinstance(idx, (list, np.ndarray)):  # 批量索引
            return [self._get_single_item(i) for i in idx]
        else:  # 单个索引
            return self._get_single_item(idx)
        # node_feature = self.node_features[idx]
        # adj_matrix = self.adj_matrices[idx]
        # label = self.labels[idx]

        # # 将数据转换为 PyTorch 张量
        # node_feature = torch.tensor(node_feature, dtype=torch.float32)
        # adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        # label = torch.tensor(label, dtype=torch.long)

        # # 应用变换（如图归一化、特征变换等）
        # if self.transform:
        #     node_feature, adj_matrix = self.transform(node_feature, adj_matrix)

        # # 将邻接矩阵转换为 edge_index 格式
        # edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
        # edge_attr = adj_matrix[adj_matrix > 0].view(-1)

        # return Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr, y=label)
    
    def _get_single_item(self, idx):
        """
        获取单个图的数据
        :param idx: 索引
        """
        node_feature = self.node_features[idx]
        adj_matrix = self.adj_matrices[idx]
        label = self.labels[idx]

        # 将数据转换为 PyTorch 张量
        node_feature = torch.tensor(node_feature, dtype=torch.float32)
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        # 应用变换（如图归一化、特征变换等）
        if self.transform:
            node_feature, adj_matrix = self.transform(node_feature, adj_matrix)

        # 将邻接矩阵转换为 edge_index 格式
        edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
        edge_attr = adj_matrix[adj_matrix > 0].view(-1)

        return Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr, y=label)


if __name__ == '__main__':
    cfpp_datas, cfpp_location = get_location('./cfpp_gnn_model/shandong_0.835.csv')
    labels = DBSCAN_cluster(cfpp_location)
    clusters, data_clusters = get_object(cfpp_datas, cfpp_location, labels)
    print(len(clusters))
    print(clusters[0])
    # print(data_clusters[0].shape)
    # print(data_clusters[6].shape)
    generate_graph_dataset(clusters, data_clusters)

    # # 测试构建的dataset
    # # 数据路径
    # x_path = './cfpp_gnn_model/merge_x.npy'
    # adj_path = './cfpp_gnn_model/merge_adj.npy'
    # y_path = './cfpp_gnn_model/merge_y.npy'

    # # 加载数据集
    # graph_dataset = GraphDataset(x_path=x_path, adj_path=adj_path, y_path=y_path)

    # # 定义 DataLoader
    # batch_size = 32
    # data_loader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=True)

    # # 测试 DataLoader
    # for batch in data_loader:
    #     print(batch)
    #     print("Batch node features shape:", batch.x.shape)
    #     print("Batch edge index shape:", batch.edge_index.shape)
    #     print("Batch labels shape:", batch.y.shape)
    #     break




    # # 测试加载
    # for i in range(3):  # 打印前 3 个样本
    #     node_feature, adj_matrix, label = graph_dataset[i]
    #     print(f"Sample {i}:")
    #     print("Node Features:\n", node_feature)
    #     print("Adjacency Matrix:\n", adj_matrix)
    #     print("Label:", label)



