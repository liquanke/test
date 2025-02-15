import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 1. 读取文件并整理数据
def load_features_from_folder(folder_path,tp_or_fp):
    all_features = []
    tp_fp_labels = []  # 用于存储真假阳性标签
    # 遍历每个.npy文件
    for npy_file in os.listdir(folder_path):
        npy_file_path = os.path.join(folder_path, npy_file)
                
        # 读取npy文件
        features = np.load(npy_file_path)  # 形状为 (1, 1024)
        features = features.reshape(-1)
        # 记录类别标签和真假阳性标签
        tp_fp_label = 1 if tp_or_fp == 'TP' else 0  # TP为1，FP为0
                
        # 这里你可以选择拼接类别标签到特征中
        all_features.append(features)
        tp_fp_labels.append(tp_fp_label)  # 每个样本添加真假阳性标签
                
    # 合并所有的特征和标签
    return all_features, tp_fp_labels

# 2. 数据加载
folder_path_tp = "C:\\Users\\14398\\Desktop\\cfpp1_all"  # 填入真阳性文件夹路径
folder_path_fp = "C:\\Users\\14398\\Desktop\\cfpp0_all"  # 填入假阳性文件夹路径

# 加载数据
features_tp, tp_fp_labels_tp = load_features_from_folder(folder_path_tp,'TP') 
features_fp, tp_fp_labels_fp = load_features_from_folder(folder_path_fp,'FP')

# 合并真阳性和假阳性数据
features = np.concatenate([features_tp, features_fp], axis=0)
tp_fp_labels = np.concatenate([tp_fp_labels_tp, tp_fp_labels_fp], axis=0)

print(features.shape)

# 3. t-SNE 降维
# 注意：t-SNE通常对数据进行标准化或归一化处理，但在此我们保持原始数据（如果必要，可以加入标准化步骤）
tsne = TSNE(n_components=2, init ='pca', random_state=42)  # 2D降维
features_tsne = tsne.fit_transform(features)  # (N, 2)

# tp_fp_labels 也转换为数字标签
encoded_tp_fp_labels = np.array(tp_fp_labels)

# 颜色映射（真阳性和假阳性）
plt.figure(figsize=(10, 8))
colors = ['tab:orange', 'tab:blue']  # 真阳性 - 蓝色，假阳性 - 橙色
tp_fp_unique = [1, 0]  # 1 为 TP， 0 为 FP
label = 'cfpp'

for j, tp_fp in enumerate(tp_fp_unique):
    # 选择类别i下的真阳性和假阳性
    tp_fp_indices = (encoded_tp_fp_labels == tp_fp)
    combined_indices = tp_fp_indices

    # 绘制该类别和真假阳性的散点
    plt.scatter(features_tsne[combined_indices, 0], features_tsne[combined_indices, 1],
                marker='s', color=colors[j], label=f'{label} - {"TP" if tp_fp == 1 else "FP"}', alpha=0.7,
                s=50)

# 添加图例
plt.legend(title="cfpp TP/FP")

# 标题和标签
plt.title("t-SNE visualization of Clip features (with TP/FP separation)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.savefig("C:\\Users\\14398\\Desktop\\clip_cfpp_all.jpg")
plt.show()