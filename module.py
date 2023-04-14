import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



# イメージとラベル表示
def show_images_labels(loader, classes, net, device):
    # データローダーから最初の1セットを取得する
    for images, labels in loader:
        break
    # 表示数は50個とバッチサイズのうち小さい方
    n_size = min(len(images), 50)

    if net is not None:
      # デバイスの割り当て
      inputs = images.to(device)
      labels = labels.to(device)

      # 予測計算
      outputs = net(inputs)
      predicted = torch.max(outputs,1)[1]
      images = images.to('cpu')

    incorrect_imges = []
    # 最初のn_size個の表示
    plt.figure(figsize=(20, 15))
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        # netがNoneでない場合は、予測結果もタイトルに表示する
        if net is not None:
          predicted_name = classes[predicted[i]]
          # 正解かどうかで色分けをする
          if label_name == predicted_name:
            c = 'k'
          else:
            c = 'b'
          ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)
        # netがNoneの場合は、正解ラベルのみ表示
        else:
          ax.set_title(label_name, fontsize=20)
        # TensorをNumPyに変換
        image_np = images[i].numpy().copy()
        # 軸の順番変更 (channel, row, column) -> (row, column, channel)
        img = np.transpose(image_np, (1, 2, 0))
        # 値の範囲を[-1, 1] -> [0, 1]に戻す
        img = (img + 1)/2

        # 結果表示
        plt.imshow(img)
        ax.set_axis_off()

    plt.show()
    plt.savefig("./models/images.png")
    plt.close() 

# 正解・不正解イメージ分けて出力
def divide_show_images(loader, classes, net, device):
    incorrect_imges = []
    correct_imges = []
    # データローダーから最初の1セットを取得する
    for images, labels in loader:
        # 表示数は50個とバッチサイズのうち小さい方
        # n_size = min(len(images), 100)
        if net is not None:
          # デバイスの割り当て
          inputs = images.to(device)
          labels = labels.to(device)

          # 予測計算
          outputs = net(inputs)
          predicted = torch.max(outputs,1)[1]
          images = images.to('cpu')

        # 最初のn_size個の表示
        for i in range(len(images)):
            num_sample = 100
            # ax = plt.subplot(13, 13, i + 1)
            label_name = classes[labels[i]]
            # netがNoneでない場合は、予測結果もタイトルに表示する
            if net is not None:
              predicted_name = classes[predicted[i]]
              # 正解かどうかで色分けをする
              if label_name == predicted_name:
                c = 'k'
              else:
                c = 'b'
            # netがNoneの場合は、正解ラベルのみ表示
            else:
              # ax.set_title(label_name, fontsize=20)
              print('no GPU')
            # TensorをNumPyに変換
            image_np = images[i].numpy().copy()
            # 軸の順番変更 (channel, row, column) -> (row, column, channel)
            img = np.transpose(image_np, (1, 2, 0))
            # 値の範囲を[-1, 1] -> [0, 1]に戻す
            img = (img + 1)/2

            if (len(correct_imges) < num_sample and c=='k'):
                correct_imges.append(img)
                # print(len(incorrect_imges))
            if (len(incorrect_imges) < num_sample and c=='b'):
                incorrect_imges.append(img)
                # print(len(incorrect_imges))
            if (len(correct_imges) == num_sample and len(incorrect_imges) == num_sample):
                break
                    
    # print(incorrect_imges)
    img_show(correct_imges, "correct")
    img_show(incorrect_imges, "incorrect")

def img_show(imges, imgType):
   print(imges[4].size)
   print(type(imges[4]))
   imgFile_name = imgType + ".png"
   num_imges = len(imges)
   print("num:",num_imges)
   for i in range(num_imges): 
      ax = plt.subplot(10, 10, i+1)
      plt.imshow(imges[i])
      ax.set_axis_off()

   plt.savefig("./output_imges/" + imgFile_name)
   plt.close()


def calc_black_whiteArea(bw_image):
    image_size = bw_image.size
    whitePixels = cv2.countNonZero(bw_image)
    blackPixels = bw_image.size - whitePixels
 
    whiteAreaRatio = (whitePixels/image_size)*100#[%]
    blackAreaRatio = (blackPixels/image_size)*100#[%]
 
    print("White Area [%] : ", whiteAreaRatio)
    print("Black Area [%] : ", blackAreaRatio)