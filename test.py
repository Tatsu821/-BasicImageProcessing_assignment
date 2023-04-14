import torch
from torch.utils.data import DataLoader
import random
import os
import numpy as np
from pathlib import Path
import torchvision 
from torchvision import transforms, datasets

from vgg16_reconstruct import myVGG
from module import show_images_labels, divide_show_images


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
print(type(test_data[2][0]))

# test_loader = DataLoader(dataset = test_data, batch_size=16, shuffle=True)
# net = torch.load('./models/best_model.pth')
# # GPU設定
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# divide_show_images(test_loader, ["0", "1"], net, device)