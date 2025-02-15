import os.path as osp
import time
import torch
import os
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from models import Net3
from cfpp_graph_dataset import GraphDataset
from sklearn.model_selection import train_test_split
import numpy as np


torch.manual_seed(7)
os.makedirs('./cfpp_gnn_model/checkpoints', exist_ok=True)

DATASET_NAME = 'cfpp'
BATCH_SIZE = 20
HIDDEN_DIM = 32
EPOCHS = 200
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-5
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.1
EARLY_STOPPING_PATIENCE = 50

# 数据路径
x_path = './cfpp_gnn_model/merge_x.npy'
adj_path = './cfpp_gnn_model/merge_adj.npy'
y_path = './cfpp_gnn_model/merge_y.npy'

# 加载数据集
graph_dataset = GraphDataset(x_path=x_path, adj_path=adj_path, y_path=y_path)

input_dim = 4
num_classes = 2

# dataset = graph_dataset.shuffle()
# 数据集划分
pos_index = range(290)
neg_index = range(2060)
train_pos_index, val_pos_index = train_test_split(pos_index, test_size=0.3, random_state=42)
train_neg__index, val_neg__index = train_test_split(neg_index, test_size=0.3, random_state=42)

train_neg__index = [i + 290 for i in train_neg__index]
val_neg__index = [i + 290 for i in val_neg__index]

train_index = np.array(train_pos_index + train_neg__index)
val_index = np.array(val_pos_index + val_neg__index)


# train_data, val_data = train_test_split(graph_data, test_size=0.2, random_state=42)

train_dataset = graph_dataset[train_index]
val_dataset = graph_dataset[val_index]
test_dataset = val_dataset

# test_dataset = graph_dataset
# val_dataset = graph_dataset
# train_dataset = graph_dataset

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('cpu')

# Model, Optimizer and Loss definitions
model = Net3(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                       patience=SCHEDULER_PATIENCE,
                                                       factor=SCHEDULER_FACTOR,
                                                       verbose=True)
nll_loss = torch.nn.NLLLoss()

best_val_loss = float('inf')
best_test_acc = 0
wait = None
for epoch in range(EPOCHS):
    # Training the model
    s_time = time.time()
    train_loss = 0.
    train_corrects = 0
    model.train()
    for i, data in enumerate(train_loader):
        s = time.time()
        data = data.to(device)
        optimizer.zero_grad()
        out, loss_pool = model(data.x, data.edge_index, data.batch)
        loss_classification = nll_loss(out, data.y.view(-1))
        loss = loss_classification + 0.01 * loss_pool

        loss.backward()
        train_loss += loss.item()
        train_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
        optimizer.step()
        # print(f'{i}/{len(train_loader)}, {time.time() - s}')

    train_loss /= len(train_loader)
    train_acc = train_corrects / len(train_dataset)
    scheduler.step(train_loss)

    # Validation
    val_loss = 0.
    val_corrects = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            s = time.time()
            data = data.to(device)
            out, loss_pool = model(data.x, data.edge_index, data.batch)
            loss_classification = nll_loss(out, data.y.view(-1))
            loss = loss_classification + 0.01 * loss_pool
            val_loss += loss.item()
            val_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
            # print(f'{i}/{len(val_loader)}, {time.time() - s}')

    val_loss /= len(val_loader)
    val_acc = val_corrects / len(val_dataset)

    # Test
    test_loss = 0.
    test_corrects = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            s = time.time()
            data = data.to(device)
            out, loss_pool = model(data.x, data.edge_index, data.batch)
            loss_classification = nll_loss(out, data.y.view(-1))
            loss = loss_classification + 0.01 * loss_pool
            test_loss += loss.item()
            test_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
            # print(f'{i}/{len(val_loader)}, {time.time() - s}')

    test_loss /= len(test_loader)
    test_acc = test_corrects / len(test_dataset)

    elapse_time = time.time() - s_time
    log = '[*] Epoch: {}, Train Loss: {:.3f}, Train Acc: {:.2f}, Val Loss: {:.3f}, ' \
          'Val Acc: {:.2f}, Test Loss: {:.3f}, Test Acc: {:.2f}, Elapsed Time: {:.1f}'\
        .format(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, best_test_acc, elapse_time)
    print(log)

    # Early-Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_test_acc = test_acc
        wait = 0
        # saving the model with best validation loss
        torch.save(model.state_dict(), f'./cfpp_gnn_model/checkpoints/{DATASET_NAME}.pkl')
    else:
        wait += 1
    # early stopping
    if wait == EARLY_STOPPING_PATIENCE:
        print('======== Early stopping! ========')
        break

