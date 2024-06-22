import torch
import warnings
import mol2graph
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import TopKPooling
from torch_geometric.loader import DataLoader
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
### 加载数据
warnings.filterwarnings('ignore')
from external_test import mol2graph_external
external_data = mol2graph_external.Lsh_MolDataset()
data = mol2graph.Lsh_MolDataset()
ld = int(len(data)*0.8)
torch.manual_seed(42)
data = data.shuffle()
train_data = data[:ld]
test_data = data[ld:]
# train_data
# train_data[0]
# train_data.y.count_nonzero()
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(test_data, batch_size=64, shuffle=False)
external_loader = DataLoader(external_data, batch_size=64, shuffle=False)
class ComplexGATWithAttention(nn.Module):
    def __init__(self):
        super(ComplexGATWithAttention, self).__init__()

        self.conv1 = GATConv(in_channels=41, out_channels=64, edge_dim=10, heads=2)
        self.conv2 = GATConv(in_channels=64*2, out_channels=64,edge_dim=10, heads=2)
        self.conv3 = GATConv(in_channels=64*2, out_channels=64,edge_dim=10, heads=2)

        self.bn1 = BatchNorm(128)
        self.bn2 = BatchNorm(128)
        self.bn3 = BatchNorm(128)
        
        self.act1 = torch.nn.ReLU()
        
        self.pool1 = TopKPooling(128, ratio=0.9)
        self.pool2 = TopKPooling(128, ratio=0.9)
        self.pool3 = TopKPooling(128, ratio=0.9)
        
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = global_mean_pool(x, batch)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = global_mean_pool(x, batch)
        
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = global_mean_pool(x, batch)
        
        # x = global_mean_pool(x, batch)  # 全局平均池化
        x = x1 + x2 + x3

        x = self.fc1(x)
        x = self.act1(x)
        
        x = self.fc2(x)
        # x = torch.sigmoid(x)
        
        return x
# data[0]
model = ComplexGATWithAttention()
# model
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
def mytrain(model, data_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
        loss = criterion(out, batch.y)
        loss.backward()
        train_loss += loss
        optimizer.step() 
    train_loss /= len(data_loader)
    return train_loss
def all_metric(model, data_loader):
    model.eval()
    predict = []
    labels = []
    val_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            # batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
            loss = criterion(out, batch.y)
            val_loss += loss  # 验证集损失
            for i in out:
                predict.append(i.item())
            for j in batch.y:
                labels.append(j.item())
                
    for i in range(len(predict)):
        if predict[i] >= 0.5:
            predict[i] = 1
        else:
            predict[i] = 0 
    # print(labels)        
    acc = accuracy_score(labels, predict)
    pre = precision_score(labels, predict)
    rec = recall_score(labels, predict)
    f1 = f1_score(labels, predict)
    mcc = matthews_corrcoef(labels, predict)
    val_loss /= len(data_loader)
    return acc, pre, rec, f1, mcc, val_loss
train_loss_value = []
val_loss_value = []
best_train_mcc = 0
best_val_mcc = 0
train_mcc_stagnant_count = 0
val_mcc_stagnant_count = 0
best_model_state_dict = None 

for epoch in range(1,601):
    train_loss = mytrain(model, train_loader, optimizer, criterion) 
    train_loss_value.append(train_loss.item())
    # train_acc = all_metric(test_model, train_loader)
    acc_1, pre_1, rec_1, f1_1, mcc_1, _= all_metric(model, train_loader)
    acc_2, pre_2, rec_2, f1_2, mcc_2, val_loss = all_metric(model, val_loader)
    val_loss_value.append(val_loss.item())
    #  early stopping
    if mcc_1 > best_train_mcc:
        best_train_mcc = mcc_1
        train_mcc_stagnant_count = 0
    else:
        train_mcc_stagnant_count += 1

    if mcc_2 > best_val_mcc:
        best_val_mcc = mcc_2
        val_mcc_stagnant_count = 0
        best_model_state_dict = model.state_dict()  # 保存最佳模型参数
    else:
        val_mcc_stagnant_count += 1

    # 判断是否停止训练
    if train_mcc_stagnant_count >= 20 and val_mcc_stagnant_count >= 35:
        print("Early stopping! Training stopped.")
        print(f"Stopped in epoch {epoch}")
        break

        
    # print(f'Epoch:{epoch}, Loss:{loss:.8f}, Train Acc:{train_acc:.4f}, Test Acc:{test_acc:.4f}')
    print(f'=========================Epoch{epoch}==========================')
    print('\n')
    print(f'训练集Loss:{train_loss:.8f}, 训练集：Acc:{acc_1:.4f}, Pre:{pre_1:.4f}, Rec:{rec_1:.4f}, F1:{f1_1:.4f}, MCC:{mcc_1:.4f}')
    print(f'验证集Loss:{val_loss:.8f}, 验证集：Acc:{acc_2:.4f}, Pre:{pre_2:.4f}, Rec:{rec_2:.4f}, F1:{f1_2:.4f}, MCC:{mcc_2:.4f}')
    print('\n')

