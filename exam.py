import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import japanize_matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision 
from torchvision import models
from torchvision import transforms, datasets

from module import show_images_labels, divide_show_images
# from module_copy import show_images_labels

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

test_data = datasets.ImageFolder(test_dataset_dir, transform)

test_loader = DataLoader(dataset = test_data, batch_size=16, shuffle=True)

image, label = test_data[0]
print(len(test_data))
print(label)
print(image.shape)

#最適なモデルを呼び出す
best_model = torch.load('./models/myvgg_model.pth')

# 正解率の計算
def test_model(test_loader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():

        accs = [] # 各バッチごとの結果格納用
        label_list = []
        pred_list = []

        for batch in test_loader:
            x, t = batch
            x = x.to(device)
            target = t
            t = t.to(device)
            y = best_model(x)

            y_label = torch.argmax(y, dim=1)
            acc = torch.sum(y_label == t) * 1.0 / len(t)
            accs.append(acc)

            label = target.numpy().tolist()
            pred = torch.argmax(best_model(x), axis = 1).cpu().numpy().tolist()
            label_list += label
            pred_list += pred

    # 全体の平均を算出
    avg_acc = torch.tensor(accs).mean()
    std_acc = torch.tensor(accs).std()
    print('Accuracy: {:.1f}%'.format(avg_acc * 100))
    print('Std: {:.4f}'.format(std_acc))
    print(accuracy_score(label_list, pred_list))

    cm = confusion_matrix(label_list,pred_list)
    print(cm)

    plt.rcParams["figure.figsize"] = (12, 10)
    # plt.rcParams['font.family'] = 'Meiryo'
    sns.set(font_scale=2)
    glaph = sns.heatmap(cm, cmap='Blues', annot=True, annot_kws={'size': 30}, fmt='d')

    glaph.set( xlabel = "predict", ylabel = "label",xticklabels=["no tumor", "with tumor"],yticklabels=["no tumor", "with tumor"])
    sfig = glaph.get_figure()
    sfig.savefig('./output_imges/mixed_matrix.png',  orientation="landscape")
    plt.close()
# テストデータで結果確認
test_model(test_loader)

# show_images_labels(test_loader, ["0", "1"], best_model, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# )
divide_show_images(test_loader, ["0", "1"], best_model, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

