# 导入需要的库
import torch
import timm
import torch.nn as nn
from torch import optim
# 导入必要的库
import random

import einops
import imageio
import numpy as np
import torch
from torch.optim import adam
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

import torch.optim as optim
import torch.nn.functional as F


from sklearn.metrics import precision_score,recall_score
images_dir = "./images/train"
test_dir = './images/test'
train_transforms = transforms.Compose([

    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Grayscale(3)



])

test_transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Grayscale(3)



])
# 使用datasets.ImageFolder类创建训练集和测试集（或者验证集）的数据集对象，传入相应的目录和预处理操作
train_dataset = datasets.ImageFolder(images_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# 使用torch.utils.data.DataLoader类创建训练集和测试集（或者验证集）的数据加载器对象，传入相应的数据集对象和批量大小等参数
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True,drop_last=True)

# 加载预训练的ViT模型
# 加载预训练的ViT模型
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)



# 确定使用GPU还是CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 把模型移动到正确的设备
model = model.to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
batch_size = 64
num_epochs = 10
# 训练模型
# for epoch in range(2):  # 迭代次数
#     for images, labels in train_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
# torch.save(model.state_dict(),f'Efficient_CIF.pt')
model.load_state_dict(torch.load('Efficient_minist.pt'))
# 测试模型
correct = 0
total = 0
all_labels = []
all_predicted = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.append(labels.detach().cpu())
        all_predicted.append(predicted.detach().cpu())
all_labels = torch.cat(all_labels)
all_predicted = torch.cat(all_predicted)
accuracy = 100 * correct / total
# 计算查准率和查全率
precision = precision_score(all_labels.numpy(), all_predicted.numpy())
recall = recall_score(all_labels.numpy(), all_predicted.numpy())

print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
print(f'Precision of the model on the test set: {precision:.2f}')
print(f'Recall of the model on the test set: {recall:.2f}')


##，这个ViT模型大约有206344个参数  8轮 80%  CIF数据集 51%