import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import cdist

def class_tranformer(df_data):
    df_data_filter = df_data[df_data['Score']>0.100]
    labels = df_data_filter['Label'].values
    for i in range(len(labels)):
        if labels[i]==1: labels[i]=0
        if labels[i]==2 or labels[i]==3: labels[i]=1
        if labels[i]==4:labels[i]=2
    return df_data_filter,labels


def generate_graph_from_csv(csv_file, distance_threshold=0.003):
    """
    根据CSV文件生成图数据,用于图神经网络模型训练。
    
    参数：
    - csv_file: str, CSV文件路径
    - distance_threshold: float, 距离阈值，控制边的连接

    返回：
    - graph_data: PyTorch Geometric Data 对象
    """
    # 1. 读取CSV文件
    df = pd.read_csv(csv_file).fillna(0)
    df_data_filter, labels = class_tranformer(df)
    # print(df.dtypes)
    coordinates = df_data_filter[['Longitude','Latitude']].values
    confidences = df_data_filter['Score'].values
    class_data = df_data_filter['Class'].values
    # print(class_data)

    
    # print(coordinates)
    # 位置坐标归一化
    np.array(coordinates)
    normalized_coordinates = (np.array(coordinates) - np.min(np.array(coordinates), axis=0)) / (np.max(np.array(coordinates), axis=0) - np.min(np.array(coordinates), axis=0))
    # print(normalized_coordinates)

    # 2. 编码子部件标签
    encoder = OneHotEncoder(sparse=False)
    label_one_hot = encoder.fit_transform(labels.reshape(-1, 1))
    # print(label_one_hot)

    # 3. 生成节点特征
    node_features = np.hstack([label_one_hot, normalized_coordinates, confidences.reshape(-1, 1)])
    print(node_features[0])

    # 4. 计算节点间距离矩阵
    distance_matrix = cdist(coordinates, coordinates, metric='euclidean')

    # 5. 生成边及其权重
    edge_index = []
    edge_attr = []
    for i in range(len(node_features)):
        for j in range(len(node_features)):
            if i != j and distance_matrix[i, j] < distance_threshold:
                edge_index.append([i, j])
                weight = 1 / (1 + distance_matrix[i, j])
                edge_attr.append(weight)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # 转置为 (2, num_edges)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor(class_data, dtype=torch.long)
    # print(edge_index.shape)
    # print(edge_attr.shape)
    # print(y.shape)

    # 6. 转换为 PyTorch Geometric Data 格式
    x = torch.tensor(node_features, dtype=torch.float)
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return graph_data

if __name__ == '__main__':
    # 示例：使用生成图数据函数
    csv_file_path = "./cfpp_gnn_model/data/shandong25_0.835.csv"  # 文件路径
    graph_data = generate_graph_from_csv(csv_file_path)

    # 输出图数据的基本信息
    print("图数据节点特征维度:", graph_data.x.size())
    print("图数据边数:", graph_data.edge_index.size(1))
    print("图数据边权重样本:", graph_data.edge_attr[:5])
