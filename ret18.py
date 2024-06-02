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


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 定义图像预处理，这里只做了缩放和裁剪，您可以根据需要添加其他变换
images_dir = "./fashionImages/train"
test_dir = './fashionImages/test'
split_ratio = 0.8
device = "cuda" if torch.cuda.is_available() else "cpu"
images_list = []
for label in ["0", "1"]:
    sub_dir = os.path.join(images_dir, label)

    for file in os.listdir(sub_dir):
        if file.endswith(".png"):
            image_path = os.path.join(sub_dir, file)
            images_list.append((image_path, int(label)))
random.shuffle(images_list)
split_index = int(len(images_list) * split_ratio)
train_list = images_list[:split_index]
test_list = images_list[split_index:]
# test_list = images_list[:split_index-360]
train_transforms = transforms.Compose([


    transforms.ToTensor(),




])

test_transforms = transforms.Compose([

    transforms.ToTensor(),




])


# 使用datasets.ImageFolder类创建训练集和测试集（或者验证集）的数据集对象，传入相应的目录和预处理操作
train_dataset = datasets.ImageFolder(images_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# 使用torch.utils.data.DataLoader类创建训练集和测试集（或者验证集）的数据加载器对象，传入相应的数据集对象和批量大小等参数
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True,drop_last=True)


device = "cuda" if torch.cuda.is_available() else "cpu"




# 定义超参数
batch_size = 64
num_epochs = 8
learning_rate = 0.01
# 定义模型，这里使用预训练的ResNet18模型，修改最后一层为二分类
model = models.resnet18(pretrained=True)

model.fc = torch.nn.Linear(model.fc.in_features, 2)
# 将模型移动到GPU上
model.to(device)

# 定义损失函数，这里使用交叉熵损失
criterion = torch.nn.CrossEntropyLoss()

# 定义优化器，这里使用随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


# 定义训练循环
# for epoch in range(num_epochs):
#     # 初始化训练损失和准确率
#     train_loss = 0.0
#     train_acc = 0.0
#     # 遍历训练数据
#     for inputs, labels in train_loader:
#         # 将数据移动到GPU上
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#
#         images = inputs
#
#
#         outputs = model(images)
#         # 将输出和标签送入损失函数，得到损失值
#         loss = criterion(outputs, labels)
#
#         # 将损失值反向传播，得到梯度
#         loss.backward()
#         # 将梯度送入优化器，更新模型参数
#         optimizer.step()
#         # 清空梯度
#         optimizer.zero_grad()
#         # 累加训练损失
#         train_loss += loss.item()
#         # 计算预测的标签
#         preds = torch.argmax(outputs, dim=1)
#         # 累加训练准确率
#         train_acc += torch.sum(preds == labels).item() / batch_size
#
#     # 计算训练损失和准确率的平均值
#     train_loss = train_loss / len(train_loader)
#     train_acc = train_acc / len(train_loader)
#     # 打印训练结果
#     print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
#
# torch.save(model.state_dict(),f'ResNet18_fminist2.pt')
# # # 定义测试循环
# # # 初始化测试损失和准确率
test_loss = 0.0
test_acc = 0.0
model.load_state_dict(torch.load('ResNet18_fminist.pt'))
# 遍历测试数据
# 初始化 TP, FP, FN
TP = 0
FP = 0
FN = 0
for inputs, labels in test_loader:
    # 将数据送入模型，得到输出
    inputs = inputs.to(device)
    labels = labels.to(device)


    images = inputs


    outputs = model(images)
    # 将输出和标签送入损失函数，得到损失值
    loss = criterion(outputs, labels)
    # 将损失值反向传播，得到梯度

    test_loss += loss.item()
    # 计算预测的标签
    preds = torch.argmax(outputs, dim=1)
    TP += ((preds == 1) & (labels == 1)).sum().item()
    FP += ((preds == 1) & (labels == 0)).sum().item()
    FN += ((preds == 0) & (labels == 1)).sum().item()
    # 累加测试准确率
    test_acc += torch.sum(preds == labels).item() / batch_size

# 计算测试损失和准确率的平均值
test_loss = test_loss / len(test_loader)
test_acc = test_acc / len(test_loader)

# 计算查准率和召回率值
precision = TP / (TP + FP)
recall = TP / (TP + FN)
# 打印测试结果
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
