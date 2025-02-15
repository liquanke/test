from sklearn.model_selection import train_test_split

# 数据集划分
pos_index = range(290)
neg_index = range(2060)
train_pos_index, val_pos_index = train_test_split(pos_index, test_size=0.3, random_state=42)
train_neg__index, val_neg__index = train_test_split(neg_index, test_size=0.3, random_state=42)

train_neg__index = [i + 290 for i in train_neg__index]
val_neg__index = [i + 290 for i in val_neg__index]

train_index = train_pos_index + train_neg__index
val_index = val_pos_index + val_neg__index


print(val_index)

