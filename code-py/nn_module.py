import torch
from torch import nn

class xiaofan(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input + 1
        return output

xiaofan = xiaofan()
x = torch.tensor(1.0)
output = xiaofan(x)
print(output)