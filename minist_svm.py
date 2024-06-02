from random import random

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# 导入必要的库
import random
import torch.optim as optim
import einops
import imageio
import numpy as np
import torch
from torch.optim import adam, SGD
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import torch.nn as nn

from sklearn.metrics import precision_score,recall_score
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=64,
    channels=1,
    dim_mults=(1, 2, 2)
)
model.load_state_dict(torch.load('test.pt'))
model.to(device)
diffusion = GaussianDiffusion(
    model,
    objective='pred_noise',
    image_size=28,
    timesteps=500,  # number of steps

)
diffusion.to(device)
images_dir = "./images/train"
test_dir = './images/test'
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
train_transforms = transforms.Compose([

    transforms.ToTensor(),
    transforms.Grayscale()

    # transforms.Resize((255, 255)),
    # transforms.CenterCrop(224)

])

test_transforms = transforms.Compose([

    transforms.ToTensor(),
    transforms.Grayscale()

])

# 使用datasets.ImageFolder类创建训练集和测试集（或者验证集）的数据集对象，传入相应的目录和预处理操作
train_dataset = datasets.ImageFolder(images_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
# 使用torch.utils.data.DataLoader类创建训练集和测试集（或者验证集）的数据加载器对象，传入相应的数据集对象和批量大小等参数
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True,drop_last=True)
t = torch.randint(0, 500, (64,), device=device).long()


def loop(self, img, return_all_timesteps=False):
    batch, device = img.shape[0], self.device
    imgs = [img]
    x_start = None
    for t in tqdm(reversed(range(0, 500)), desc='sampling loop time step', total=500):
        self_cond = x_start if self.self_condition else None
        img, x_start = self.p_sample(img, t, self_cond)
        imgs.append(img)

    ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
    ret = self.unnormalize(ret)
    start = x_start.clone()
    start[start <= 0] = 1
    return ret, start

def get_mse(images):
    list = []
    images = images.to(device)
    img = torch.randn(images.shape, device=device)
    noise_img = images + img #前向过程
    pre_img, x_start = loop(diffusion, noise_img) #反向过程
    images1 = images.clone()
    start = x_start.clone()
    images1[images1 <= 0] = 1
    j = 0
    while j <= 63:
        loss_function = torch.nn.MSELoss()
        mse_value = loss_function(start[j], images1[j])
        list.append([mse_value])
        j = j + 1
    return  torch.tensor(list)


# 定义模型
class MLPBinaryClassifier(nn.Module):
    def __init__(self):
        super(MLPBinaryClassifier, self).__init__()
        self.mse = get_mse #重建误差计算
        self.fc1 = nn.Linear(1, 10)  # 输入层到隐藏层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(10, 1)  # 隐藏层到输出层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mse(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 初始化模型
model = MLPBinaryClassifier()

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器



# # 训练模型
# for epoch in range(5):
#     for data, labels in train_loader:
#
#         labels  = labels.unsqueeze(1)
#
#         outputs = model(data)  # 将一维数据扩展为二维张量
#
#         loss = criterion(outputs, labels.float())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#
#
#         print(f"Epoch [{epoch + 1}], Loss: {loss.item():.4f}")
# torch.save(model.state_dict(),f'ministMLP_{5}.pt')
# 在测试集上评估模型
# 测试模型
model.load_state_dict(torch.load(f'ministMLP_5.pt'))
def eval_model(test_loader):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        correct = 0
        total = 0
        all_labels = []
        all_predicted = []
        for features, labels in test_loader:
            outputs = model(features)
            labels = labels.unsqueeze(1)
            predicted = (outputs.data > 0.5).float()
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
# 计算混淆矩阵
    cm = confusion_matrix(all_labels.numpy(), all_predicted.numpy())

    # 使用 matplotlib 和 seaborn 来绘制混淆矩阵
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
eval_model(test_loader)