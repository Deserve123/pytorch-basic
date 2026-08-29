import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1  模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2  模型参数
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


class xiaofan(nn.Module):
    def __init__(self):
        super(xiaofan, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

xiaofan_model = xiaofan()
torch.save(xiaofan_model, "xiaofan_method1.pth")