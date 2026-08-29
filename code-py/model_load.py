import torch
from model_save import *

import torchvision
from torch import nn

# 方式1 加载方式一
model = torch.load("vgg16_method1.pth")


# 方式2 加载方式二
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))


# 陷阱
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

model = torch.load('xiaofan_method1.pth')
print(model)