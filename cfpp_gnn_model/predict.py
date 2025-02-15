import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from models import Net3
from cfpp_graph_dataset import GraphDataset
import pandas as pd
from sklearn.model_selection import train_test_split

# 数据路径
x_path = './cfpp_gnn_model/merge_x.npy'
adj_path = './cfpp_gnn_model/merge_adj.npy'
y_path = './cfpp_gnn_model/merge_y.npy'

all_x_path = './cfpp_gnn_model/all_x.npy'
all_adj_path = './cfpp_gnn_model/all_adj.npy'

# 测试集加载
graph_dataset = GraphDataset(x_path=x_path, adj_path=adj_path, y_path=y_path)

# 数据集划分
pos_index = range(290)
neg_index = range(2060)
train_pos_index, val_pos_index = train_test_split(pos_index, test_size=0.3, random_state=42)
train_neg__index, val_neg__index = train_test_split(neg_index, test_size=0.3, random_state=42)

train_neg__index = [i + 290 for i in train_neg__index]
val_neg__index = [i + 290 for i in val_neg__index]

train_index = np.array(train_pos_index + train_neg__index)
val_index = np.array(val_pos_index + val_neg__index)
val_dataset = graph_dataset[val_index]

test_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)

# 预测集加载
predict_dataset = GraphDataset(x_path=all_x_path, adj_path=all_adj_path, y_path=None)
predict_loader = DataLoader(predict_dataset, batch_size=20, shuffle=False)


# 模型定义
input_dim = 4
hidden_dim = 32
num_classes = 2
device = torch.device('cpu')  # 根据实际情况调整为 'cuda' 或 'cpu'

model = Net3(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device)

# 加载权重文件
checkpoint_path = './cfpp_gnn_model/checkpoints/cfpp.pkl'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# 预测和评估
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out, _ = model(data.x, data.edge_index, data.batch)
        probs = torch.softmax(out, dim=1)  # 转为概率
        preds = probs.argmax(dim=1)  # 获取预测类别

        all_probs.extend(probs.cpu().numpy())  # 预测概率
        all_preds.extend(preds.cpu().numpy())  # 预测类别
        all_labels.extend(data.y.view(-1).cpu().numpy())  # 真实标签

# 转换为 NumPy 数组
all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

result_predict = np.append(all_probs,all_preds.reshape(-1, 1),axis=1)
result_predict = np.append(result_predict,all_labels.reshape(-1, 1),axis=1)

# print(all_probs)
# print(all_preds)
# print(all_labels)
# 保存预测结果
# result_predict = pd.DataFrame(result_predict, columns=['probs0','probs1', 'preds', 'labels'])
# result_predict.to_csv('./cfpp_gnn_model/result_predict.csv')

# 打印二分类指标
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))


# 预测集预测
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for data in predict_loader:
        data = data.to(device)
        out, _ = model(data.x, data.edge_index, data.batch)
        probs = torch.softmax(out, dim=1)  # 转为概率
        preds = probs.argmax(dim=1)  # 获取预测类别

        all_probs.extend(probs.cpu().numpy())  # 预测概率
        all_preds.extend(preds.cpu().numpy())  # 预测类别

# 转换为 NumPy 数组
all_probs = np.array(all_probs)
all_preds = np.array(all_preds)

result_predict = np.append(all_probs,all_preds.reshape(-1, 1),axis=1)

# print(all_probs)
# print(all_preds)
# print(all_labels)
# 保存预测结果
# result_predict = pd.DataFrame(result_predict, columns=['probs0','probs1', 'preds'])
# result_predict.to_csv('./cfpp_gnn_model/all_result_predict.csv')