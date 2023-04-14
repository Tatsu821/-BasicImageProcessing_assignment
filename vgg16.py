import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision 
from torchvision import models
from torchvision import transforms, datasets

from vgg16_reconstruct import myVGG

##########データの準備##########
seed= 11
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

train_dataset_dir = Path('Dataset/train')
val_dataset_dir = Path('Dataset/val')
test_dataset_dir = Path('Dataset/test')

transform = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Lambda(lambda x: x.view(-1))
    ]
)

train_data = datasets.ImageFolder(train_dataset_dir,transform)
valid_data = datasets.ImageFolder(val_dataset_dir, transform)
test_data = datasets.ImageFolder(test_dataset_dir, transform)

train_loader = DataLoader(dataset = train_data, batch_size=16, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=16, shuffle=False)
test_loader = DataLoader(dataset = test_data, batch_size=16, shuffle=False)



image, label = train_data[8000]
print(len(train_data))
print(label)
print(image.shape)


##########モデルの学習##########

#GPU設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

#モデルの定義
net = models.vgg16(pretrained=False)
# net = myVGG()

print(net)

net.to(device)
net.train()

##########モデルの学習##########
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.00001)

#訓練・検証データの正答率と損失リスト
train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []

# 同じデータをnumEpoch回学習します
numEpoch = 23
for epoch in range(numEpoch):
  print('Epoch {}/{}'.format(epoch + 1, numEpoch))
  print('-'*20)

  # 今回の学習効果を保存するための変数
  running_loss = 0.0
  epoch_loss = 0
  epoch_accuracy = 0

  for data in tqdm(train_loader):
    # データ整理
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    # 前回の勾配情報をリセット
    optimizer.zero_grad()

    # 予測
    outputs = net(inputs)

    # 予測結果と教師ラベルを比べて損失を計算
    loss = criterion(outputs, labels)
    acc = (outputs.argmax(dim=1) == labels).float().sum()
    epoch_accuracy += acc /len(train_data)
    epoch_loss += loss /len(train_data)
    running_loss += loss.item()

    # 損失に基づいてネットワークのパラメーターを更新
    loss.backward()
    optimizer.step()
  
  train_acc_list.append(epoch_accuracy)
  train_loss_list.append(epoch_loss)

  # このエポックの学習効果
  print(running_loss)


  # optimizer = optim.Adam(net.parameters(), lr=0.000001)

# for epoch in range(numEpoch): 
#   print('Epoch {}/{}'.format(epoch + 1, numEpoch))
#   print('-'*20)

  #学習モード切替
  # net.train()

  running_loss = 0.0
  val_running_loss = 0.0
  epoch_val_accuracy = 0
  epoch_val_loss = 0

  # for data in tqdm(train_loader):
  #   inputs, labels = data
  #   inputs = inputs.to(device)
  #   labels = labels.to(device)
  #   optimizer.zero_grad()
  #   outputs = net(inputs)
  #   loss = criterion(outputs, labels)
  #   running_loss += loss.item()

  #   loss.backward()
  #   optimizer.step()

  # モデルを評価モードにする
  net.eval()

  # 全検証データの正しく分類できた枚数を記録
  n_correct = 0
  n_total = 0
  with torch.no_grad():
    for data in tqdm(valid_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 予測
        outputs = net(inputs)
        val_loss = criterion(outputs, labels)
        val_running_loss += val_loss.item()

        # 予測結果をクラス番号に変換
        _, predicted = torch.max(outputs.data, 1)
        
        # 予測結果と実際のラベルを比較して、正しく予測できた枚数を計算
        res = (predicted == labels)
        res = res.sum().item()

        acc = (outputs.argmax(dim=1) == labels).float().sum()

        # 今までに正しく予測できた枚数に計上
        n_correct = n_correct + res
        n_total = n_total + len(labels)

        epoch_val_accuracy += acc / len(valid_data)
        epoch_val_loss += val_loss / len(valid_data)

    val_acc_list.append(epoch_val_accuracy)
    val_loss_list.append(epoch_val_loss)


    print(running_loss)
    print(val_running_loss)
    print(n_correct / n_total)

torch.save(net, './models/best_model.pth')
print(train_loss_list)
print(val_loss_list)








# ##########モデルの検証##########
#   # モデルを評価モードにする
# net.eval()

# # 全検証データの正しく分類できた枚数を記録
# n_correct = 0
# n_total = 0

# for data in valid_loader:
#     inputs, labels = data
#     inputs = inputs.to(device)
#     labels = labels.to(device)

#     # 予測
#     outputs = net(inputs)
#     val_loss = criterion(outputs, labels)

#     # 予測結果をクラス番号に変換
#     _, predicted = torch.max(outputs.data, 1)
    
#     # 予測結果と実際のラベルを比較して、正しく予測できた枚数を計算
#     res = (predicted == labels)
#     res = res.sum().item()

#     acc = (outputs.argmax(dim=1) == labels).float().mean()
#     

#     # 今までに正しく予測できた枚数に計上
#     n_correct = n_correct + res
#     n_total = n_total + len(labels)




# torch.save(net, 'best_model.pth')


# print(n_correct / n_total)

device2 = torch.device('cpu')

train_acc = []
train_loss = []
val_acc = []
val_loss = []

for i in range(numEpoch):
    train_acc2 = train_acc_list[i].to(device2)
    train_acc3 = train_acc2.clone().numpy()
    train_acc.append(train_acc3)
    
    train_loss2 = train_loss_list[i].to(device2)
    train_loss3 = train_loss2.clone().detach().numpy()
    train_loss.append(train_loss3)
    
    val_acc2 = val_acc_list[i].to(device2)
    val_acc3 = val_acc2.clone().numpy()
    val_acc.append(val_acc3)
    
    val_loss2 = val_loss_list[i].to(device2)
    val_loss3 = val_loss2.clone().numpy()
    val_loss.append(val_loss3)
print(val_loss)
print(train_loss)

#取得したデータをグラフ化する
sns.set()
num_epochs = numEpoch

fig = plt.subplots(figsize=(12, 4), dpi=80)

ax1 = plt.subplot(1,2,1)
ax1.plot(range(num_epochs), train_acc, c='b', label='train acc')
ax1.plot(range(num_epochs), val_acc, c='r', label='val acc')
ax1.set_xlabel('epoch', fontsize='12')
ax1.set_ylabel('accuracy', fontsize='12')
ax1.set_title('training and val acc', fontsize='14')
ax1.legend(fontsize='12')

ax2 = plt.subplot(1,2,2)
ax2.plot(range(num_epochs), train_loss, c='b', label='train loss')
ax2.plot(range(num_epochs), val_loss, c='r', label='val loss')
ax2.set_xlabel('epoch', fontsize='12')
ax2.set_ylabel('loss', fontsize='12')
ax2.set_title('training and val loss', fontsize='14')
ax2.legend(fontsize='12')


plt.savefig('./output_imges/fig.png')
