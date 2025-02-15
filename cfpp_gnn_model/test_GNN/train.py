from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from generate_graph import generate_graph_from_csv
from GNN import GNNNodeClassifier
import pandas as pd

# 图数据
csv_file_path = "./cfpp_gnn_model/data/shandong25_0.835.csv"  # 文件路径
graph_data = generate_graph_from_csv(csv_file_path)

# 数据集划分
# train_data, val_data = train_test_split(graph_data, test_size=0.2, random_state=42)

# 模型实例化
input_dim = graph_data.x.shape[1]
hidden_dim = 16
output_dim = 1  # 二分类任务
model = GNNNodeClassifier(input_dim, hidden_dim, output_dim)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

# 模型训练
num_epochs = 60
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    loss = criterion(out.squeeze(), graph_data.y.float())
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# 模型验证
model.eval()
with torch.no_grad():
    val_out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    val_loss = criterion(val_out.squeeze(), graph_data.y.float())
    print(f"Validation Loss: {val_loss.item():.4f}")
    print(val_out.numpy().reshape(-1))
    result_pd = pd.DataFrame(val_out.numpy().reshape(-1),columns=['reslut'])
    result_pd.to_csv('result.csv')


