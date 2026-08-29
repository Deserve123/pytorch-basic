# PyTorch 基础学习课程总结

---

# 第一板块：课程入门

## 1.1 课程定位

学的是 **PyTorch 深度学习入门框架**，以「卷积神经网络 CNN 做图像分类」为贯穿案例，把"搭网络 → 喂数据 → 训练 → 保存 → 推理"整条链路跑通。

## 1.2 核心能力

| 能力     | 对应代码                                                      | 解决什么          |
| ------ | --------------------------------------------------------- | ------------- |
| 定义网络结构 | `nn_module` / `nn_seq` / `model`                          | 让模型"长什么样"     |
| 提取图像特征 | `nn_conv2d` / `nn_maxpool` / `nn_relu`                    | 让模型"看懂图"      |
| 输出分类结果 | `nn_linear`                                               | 让模型"做决策"      |
| 量化误差   | `nn_loss` / `nn_loss_network`                             | 知道"差多少"       |
| 更新参数   | `nn_optim`                                                | 知道"怎么改"       |
| 准备数据   | `dataset` / `dataloader` / `Transforms`                   | 把图片变成能喂的张量    |
| 监控训练   | `Tensorboard`                                             | 看到 loss/准确率曲线 |
| 复用/落地  | `model_save` / `model_load` / `model_pretrained` / `test` | 保存、迁移、推理      |

## 1.3 使用场景

- **图像分类**：猫狗识别、商品分类、缺陷检测（最常见入门场景）。
- **迁移学习**：拿 VGG/ResNet 预训练权重改几层，小数据也能高精度（`model_pretrained`）。
- **模型上线**：训练完 `torch.save`，部署端 `torch.load` + `no_grad` 推理（`test`）。
- **科研/教学原型**：Jupyter 交互式调试（`demo_jupyter` 的 `# %%` cell）。

---

# 第二板块：环境配置

## 2.1 安装清单

| 组件          | 说明                                                                                                                        |
| ----------- | --------------------------------------------------------------------------------------------------------------------           |
| Python 3.12 | python运行环境                                                                                                                 |
| PyTorch     | pip install torch torchvision torchaudio （CPU版）        pytorch.org 复制带 cuda 的命令（GPU版）                               |
| Anaconda    | Anaconda/Miniconda        （完整版和轻量版）                                                                                    |
| VSCode      | python  jupyter(交互式运行工具)  vscode-icons(文件图标)                                                                         |

## 2.2 VSCode 使用要点

- **选对解释器**：`Ctrl+Shift+P` → `Python: Select Interpreter` → 选装了 torch 的环境。
- **交互式运行**：`demo_jupyter.py` 用 `# %%` 把脚本切成 cell，点右上角 ▶ 逐格运行，适合验证张量形状。
- **GPU 自检**：`demo_jupyter` 首格就打印版本与 `cuda.is_available()`，装完先跑它确认环境对

```python
import torch
print(torch.__version__, torch.cuda.is_available())   # 先看这俩再写代码
```

## 2.3 注意踩坑

代码里数据路径是相对路径（`"dataset"`、`"dataself/train"`）。**插件在 `code-py` 目录跑 `.py` 时，当前目录是 `code-py`，会找不到平级的 `data` 文件夹**。

`demo_jupyter.py` 给出的标准解法：

```python
import os
os.chdir(r'D:\wyf\study\python\pytorch-test')  # 把 cwd 钉到项目根目录
print(os.getcwd())
```

## 2.4 数据来源

- **CIFAR10**
- **本地数据**

---

# 第三板块：学习内容

## 3.1 nn_module.py 网络基石：Module 与 forward

**1. 初步讲解**

- `torch.nn.Module` 是 PyTorch 中所有神经网络模块的基类，自定义网络必须继承它。
- `__init__` 中调用 `super().__init__()` 完成父类的初始化，把子层注册到模块中。
- `forward(self, x)` 定义前向传播逻辑，决定输入张量如何被逐层计算得到输出。
- 调用模型实例 `model(x)` 时，PyTorch 内部会自动触发 `forward`，而不是手动调用 `forward()`。

**2. 核心代码**

```python
import torch
from torch import nn

class xiaofan(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1   # 最简前向：每个元素 +1
        return output

xiaofan = xiaofan()
x = torch.tensor(1.0)
output = xiaofan(x)         # 自动调用 forward
print(output)               # tensor(2.)
```

**3. 落地场景**

- 所有自定义网络（CNN、RNN、Transformer）都从 `nn.Module` 派生，是写的第一个类。
- 把网络结构写在 `__init__`，把数据流向写在 `forward`，方便复用与组合。
- 借助 `Module` 自动管理的参数（`parameters()`）才能被优化器更新。
- 调试时把 `forward` 当成普通函数加打印，即可观察每一层张量形状。

**4. 最终复盘**

- **易出问题**：
  1. 忘记写 `super().__init__()`，导致子层参数无法被 PyTorch 跟踪。
  2. 直接调用 `model.forward(x)` 而不是 `model(x)`，丢失钩子与自动求导。
  3. `__init__` 里新建张量却没注册为子模块，导致它不在 `parameters()` 中。
- **解决方法**：
  1. 在 `__init__` 首行固定写 `super().__init__()`。
  2. 始终用 `output = model(x)` 触发前向传播。
  3. 需要可学习参数时用 `nn.Parameter` 或 `nn.Linear` 等子模块承载。
- **踩坑原因**：根因是 `nn.Module` 依赖 `super().__init__()` 初始化内部 `_parameters`、`_modules` 等寄存器，漏写会使子层脱离自动管理机制，从而无法被优化器识别与保存。
- **更优解**：
  1. 把可复用子层放进 `nn.Sequential`，减少手写 `forward` 样板代码。
  2. 用 `self.children()` / `self.modules()` 遍历子层做统一初始化或冻结。

## 3.2 nn_conv2d.py 卷积层：提取空间特征

**1. 初步讲解**

- `nn.Conv2d` 是二维卷积层，用滑动窗口（卷积核）在特征图上提取局部空间特征。
- 关键参数：`in_channels`（输入通道）、`out_channels`（输出通道）、`kernel_size`（核大小）、`stride`（步长）、`padding`（填充）。
- 卷积能在保留空间结构的同时减少参数量，是 CNN 的核心特征提取器。
- 输出尺寸公式：`out = (in - kernel + 2*padding) / stride + 1`（向下取整）。

**2. 核心代码**

```python
import torch
from torch.nn import Conv2d

class xiaofan(nn.Module):
    def __init__(self):
        super(xiaofan, self).__init__()
        # 输入3通道, 输出6通道, 核3x3, 步长1, 不填充
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

xiaofan = xiaofan()
imgs = torch.ones((64, 3, 32, 32))   # (N, C, H, W)
output = xiaofan(imgs)
print(imgs.shape)    # torch.Size([64, 3, 32, 32])
print(output.shape)  # torch.Size([64, 6, 30, 30])
# 公式验证: (32-3+0)/1+1 = 30
```

**3. 落地场景**

- 图像分类、检测、分割等任务的第一组层几乎都是 `Conv2d` 堆叠。
- 通过多层卷积逐步扩大感受野、抽象出边缘→纹理→语义的高级特征。
- 第一层 `in_channels=3`（RGB），深层通道数成倍增长以承载更丰富特征。
- 配合 `padding` 控制特征图尺寸，常取 `padding=kernel//2` 保持分辨率。

**4. 最终复盘**

- **易出问题**：
  1. 维度不匹配：输入不是 `(N, C, H, W)` 四维张量。
  2. padding 算错导致输出尺寸与后续 `Linear` 输入对不上。
  3. 把 `kernel_size` 写成整数却想用非方形核，容易混淆。
- **解决方法**：
  1. 输入先用 `x.unsqueeze(0)` 补 batch 维，或确保 dataloader 已加 batch 维。
  2. 用公式 `out=(in-kernel+2p)/s+1` 提前算好尺寸再设计网络。
  3. 非方形核显式写成元组 `kernel_size=(3,5)`。
- **踩坑原因**：根因是卷积运算要求输入严格为四维 `(batch, channel, height, width)`，且输出尺寸由卷积公式精确决定，任何维度或 padding 的偏差都会沿网络向后累积放大。
- **更优解**：
  1. 优先用 `padding=kernel_size//2` 让特征图尺寸随 stride 整除变化，省去手算。
  2. 用 `nn.Conv2d(...).to(device)` 统一管理设备，避免 CPU/GPU 张量混合。

## 3.3 nn_maxpool.py 池化层：降维抗过拟合

**1. 初步讲解**

- `nn.MaxPool2d` 在局部窗口取最大值，实现下采样，缩小特征图尺寸。
- 主要作用：减少计算量与参数、提供平移不变性、抑制噪声、缓解过拟合。
- 关键参数：`kernel_size`（窗口）、`stride`（默认等于 kernel_size）、`ceil_mode`（是否用向上取整）。
- 池化层无可学习参数，只做固定规则运算。

**2. 核心代码**

```python
from torch.nn import MaxPool2d

class xiaofan(nn.Module):
    def __init__(self):
        super(xiaofan, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)  # 窗口3x3

    def forward(self, input):
        output = self.maxpool1(input)
        return output
# 配合 TensorBoard 可观察 input 与 output 特征图的变化
```

**3. 落地场景**

- 通常穿插在卷积层之间，每经过一次池化特征图尺寸减半。
- 在 3.6 的 `model.py` 中每个 `Conv2d` 后接 `MaxPool2d(2)`，把 32x32 压到 4x4。
- 减少后续全连接层输入维度，控制模型规模。
- 增强模型对微小位移的鲁棒性，提高泛化能力。

**4. 最终复盘**

- **易出问题**：
  1. `ceil_mode=True` 时输出尺寸向上取整，与预期不符。
  2. 池化次数过多导致特征图被压成 0 维，后续层报错。
  3. 误以为池化层有参数需要优化。
- **解决方法**：
  1. 默认用 `ceil_mode=False`，或按公式预计算输出尺寸。
  2. 在堆叠池化前打印各层 shape，确保不低于 1x1。
  3. 记住池化无参数，`model.parameters()` 中不会包含它。
- **踩坑原因**：根因是池化窗口滑动的步长与取整方式共同决定输出尺寸，当特征图较小或 `ceil_mode` 开启时尺寸计算会偏离直觉，进而引发后续维度断裂。
- **更优解**：
  1. 现代网络常用 stride=2 的卷积替代池化来做下采样，保留更多信息。
  2. 需要保留精确位置时用 `nn.AvgPool2d` 替代最大值池化。

## 3.4 nn_relu.py 激活函数：引入非线性

**1. 初步讲解**

- 激活函数为网络引入非线性，使多层网络能拟合复杂函数。
- `nn.ReLU` 把负值截断为 0：`f(x)=max(0, x)`，缓解梯度消失、计算快。
- `nn.Sigmoid` 把输出压到 (0,1)，多用于二分类或门控，但易梯度消失。
- 无激活函数的纯线性层叠加等价于单层线性变换，失去深度意义。

**2. 核心代码**

```python
from torch.nn import ReLU, Sigmoid

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))

class xiaofan(nn.Module):
    def __init__(self):
        super(xiaofan, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)   # 示例用 sigmoid
        return output
# ReLU 后: [[1,0],[0,3]]; Sigmoid 后: 每个元素映射到 (0,1)
```

**3. 落地场景**

- 卷积/线性层之后通常接 `ReLU`，构成 `Conv->BN->ReLU` 经典组合。
- 分类网络最后一层视任务选择：`Sigmoid`（二分类）或 `Softmax`（多分类，常藏在 `CrossEntropyLoss` 内）。
- 用 TensorBoard 可视化激活前后特征图，观察 ReLU 如何稀疏化特征。
- 调试时可临时替换为 `ReLU` 对比 `Sigmoid` 的收敛速度差异。

**4. 最终复盘**

- **易出问题**：
  1. 忘记加激活函数，网络退化为线性，无法拟合复杂分布。
  2. 深层网络用 `Sigmoid` 导致梯度消失，损失几乎不下降。
  3. 输入未归一化时 `ReLU` 死神经元（输出恒 0）。
- **解决方法**：
  1. 每层非线性变换后紧跟激活函数，形成标准组合。
  2. 默认用 `ReLU`/`LeakyReLU`，避免深层 `Sigmoid`。
  3. 对输入做标准化（如 `ToTensor` + 均值方差），缓解死神经元。
- **踩坑原因**：根因是激活函数负责打破层间线性叠加，缺失或用易饱和函数会让反向传播梯度趋零，使深层参数无法得到有效更新。
- **更优解**：
  1. 用 `nn.LeakyReLU`/`nn.PReLU` 缓解死神经元问题。
  2. 配合 `BatchNorm2d` 稳定激活值分布，提升训练稳定性。

## 3.5 nn_linear.py 线性层：分类决策头

**1. 初步讲解**

- `nn.Linear` 实现全连接变换：`y = xA^T + b`，对特征做线性组合。
- 常用于网络末端，把展平后的特征映射到类别分数（logits）。
- 参数：`in_features`（输入维度）、`out_features`（输出维度）。
- 输入需是二维 `(N, in_features)`，高维特征要先 `Flatten` 或 `reshape`。

**2. 核心代码**

```python
from torch.nn import Linear

class xiaofan(nn.Module):
    def __init__(self):
        super(xiaofan, self).__init__()
        self.linear1 = Linear(196608, 10)   # 输入展平维度 -> 10类

    def forward(self, input):
        output = self.linear1(input)
        return output

# 展平示例: (64,3,32,32) -> (64, 196608)
imgs = torch.ones((64, 3, 32, 32))
output = torch.flatten(imgs, 1)   # 展平除 batch 维: (64, 196608)
```

**3. 落地场景**

- CNN 末尾 `Flatten` 后用 `Linear` 输出类别数（如 CIFAR10 的 10）。
- 在 3.6 的 `model.py` 中两道 `Linear(64*4*4, 64)`、`Linear(64, 10)` 构成分类头。
- 回归任务把 `out_features` 设为 1 或目标维度。
- 调试维度时先 `print(x.shape)` 再确定 `in_features`。

**4. 最终复盘**

- **易出问题**：
  1. `in_features` 算错，与展平后维度对不上报 size mismatch。
  2. 忘记 `Flatten`，四维张量直接进 `Linear` 报错。
  3. 展平时把 batch 维一起展平，破坏样本独立性。
- **解决方法**：
  1. 用 `x.flatten(1).shape[-1]` 动态获取 `in_features` 再定义层。
  2. 在 `Linear` 前加 `nn.Flatten()` 层自动处理。
  3. 展平只针对特征维：`torch.flatten(x, 1)` 保留第 0 维 batch。
- **踩坑原因**：根因是 `Linear` 只接受二维 `(N, D)` 输入且要求 `D` 与 `in_features` 严格相等，展平方式或维度计算错误会直接引发矩阵乘法维度不匹配。
- **更优解**：
  1. 用 `nn.AdaptiveAvgPool2d(1)` 替代手动展平，自适应输出维度更稳健。
  2. 分类头前加 `Dropout` 降低过拟合风险。

## 3.6 nn_seq.py / model.py 容器 Sequential：组装完整网络

**1. 初步讲解**

- `nn.Sequential` 是一个有序容器，按顺序把多个子层串成一条前向流水线。
- 在 `forward` 中只需 `x = self.model(x)` 即可走完整条链路。
- `model.py` 把卷积、池化、展平、线性组合成一个可直接训练的完整网络。
- 用固定 padding 让卷积后尺寸可预测，便于衔接 `Linear`。

**2. 核心代码**

```python
# model.py —— 完整 CIFAR10 分类网络
class xiaofan(nn.Module):
    def __init__(self):
        super(xiaofan, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),   # 32x32 -> 32x32
            nn.MaxPool2d(2),             # 32x32 -> 16x16
            nn.Conv2d(32, 32, 5, 1, 2),  # 16x16 -> 16x16
            nn.MaxPool2d(2),             # 16x16 -> 8x8
            nn.Conv2d(32, 64, 5, 1, 2),  # 8x8  -> 8x8
            nn.MaxPool2d(2),             # 8x8  -> 4x4
            nn.Flatten(),                # (64,64,4,4) -> (64, 64*4*4)
            nn.Linear(64*4*4, 64),       # 1024 -> 64
            nn.Linear(64, 10)            # 64 -> 10 类
        )

    def forward(self, x):
        x = self.model(x)
        return x

xiaofan_model = xiaofan()
input = torch.ones((64, 3, 32, 32))
print(xiaofan_model(input).shape)   # torch.Size([64, 10])
```

**3. 落地场景**

- 把 3.2~3.5 学到的各层像搭积木一样组装成可训练模型。
- `nn_seq.py` 用 `padding=2`（kernel=5）保持尺寸，使每步尺寸可控。
- 整个网络作为参数整体交给优化器，配合 3.7 损失、3.8 优化器完成训练。
- 通过 `print(model)` 查看结构，用 3.10 的 `add_graph` 可视化计算图。

**4. 最终复盘**

- **易出问题**：
  1. `Linear` 输入维度 `64*4*4` 与前面卷积池化实际输出不符。
  2. `Sequential` 内某层输出形状不能被下一层接受。
  3. 卷积 `padding` 设置不当导致特征图尺寸在多次池化后变 0。
- **解决方法**：
  1. 用 `model(torch.ones(1,3,32,32)).shape` 实跑一次验证最终维度。
  2. 逐层插入 `print` 或断点查看中间 shape。
  3. 统一用 `padding=kernel//2` 配合 `stride=1` 保持分辨率。
- **踩坑原因**：根因是 `Sequential` 仅按层序直连，前层输出维度必须等于后层输入维度，任何一处卷积/池化尺寸估算错误都会在后层 `Linear` 处集中爆发。
- **更优解**：
  1. 复杂分支结构用 `nn.Module` 手写 `forward` 代替 `Sequential`。
  2. 用 `nn.Flatten()` 与自适应池化避免手写 `64*4*4` 这类易错常量。

## 3.7 nn_loss.py / nn_loss_network.py 损失函数：衡量预测好坏

**1. 初步讲解**

- 损失函数计算模型输出与真实标签的差距，是优化的目标信号。
- `L1Loss`（MAE）取绝对差，`MSELoss`（MSE）取平方差，多用于回归。
- `CrossEntropyLoss` 内置 `Softmax`，输入 raw logits、标签为类别索引，用于多分类。
- 损失值越小代表预测越接近真实，反向传播即沿损失梯度下降。

**2. 核心代码**

```python
# nn_loss.py —— 基础损失
inputs  = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)
inputs  = torch.reshape(inputs,  (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction='sum')
result = loss(inputs, targets)        # 绝对差之和 = |0|+|0|+|2| = 2

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets) # 平方差 = 0+0+4 = 4

x = torch.tensor([0.1, 0.2, 0.3])      # 3类 logits
y = torch.tensor([1])                  # 真实类别索引
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)        # 内置 softmax 后的交叉熵
```

**3. 落地场景**

- 分类任务统一用 `nn.CrossEntropyLoss()`，标签直接传类别下标不用 one-hot。
- 在 3.13 训练循环中 `loss = loss_fn(outputs, targets)` 计算每批损失。
- 回归任务（如坐标、价格）改用 `MSELoss`/`L1Loss`。
- 多标签分类可用 `BCEWithLogitsLoss` 替代。

**4. 最终复盘**

- **易出问题**：
  1. `CrossEntropyLoss` 输入已带 `Softmax` 又手动 softmax，导致数值错误。
  2. 标签传成 one-hot 向量而非类别索引，维度不匹配报错。
  3. 形状 `(N,C,H,W)` 与 `(N,C)` 不匹配，要求 loss 输入输出形状一致。
- **解决方法**：
  1. 用 `CrossEntropyLoss` 时模型末尾不要加 `Softmax`。
  2. 分类标签用整数索引 `targets`（如 `torch.tensor([1])`）。
  3. 回归损失前把标签 `reshape` 到与预测相同形状。
- **踩坑原因**：根因是各损失函数对输入格式有严格约定（`CrossEntropyLoss` 期望 raw logits 加类别索引），格式不符会在内部 softmax/维度对齐时报错或得到错误梯度。
- **更优解**：
  1. 类别不平衡时给 `CrossEntropyLoss(weight=...)` 加权。
  2. 用 `label_smoothing` 参数缓解过拟合、提升泛化。

## 3.8 nn_optim.py 优化器：反向传播更新参数

**1. 初步讲解**

- 优化器根据损失反向传播的梯度更新模型参数，是训练的执行者。
- `torch.optim.SGD` 随机梯度下降，`lr` 控制步长（学习率）。
- 标准三步：`zero_grad()` 清零梯度 → `loss.backward()` 反向传播 → `step()` 更新参数。
- `StepLR` 等学习率调度器可按轮数衰减 `lr`，帮助后期收敛。

**2. 核心代码**

```python
# nn_optim.py —— 优化器与调度器
loss = nn.CrossEntropyLoss()
xiaofan = xiaofan()
optim = torch.optim.SGD(xiaofan.parameters(), lr=0.01)
scheduler = StepLR(optim, step_size=5, gamma=0.1)   # 每5轮 lr 乘0.1

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = xiaofan(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()       # 1. 梯度清零
        result_loss.backward()  # 2. 反向传播求梯度
        optim.step()            # 3. 更新参数
        scheduler.step()        # 学习率调度
        running_loss += result_loss
```

**3. 落地场景**

- 在 3.13 完整训练里每个 batch 都执行 `zero_grad→backward→step` 三步。
- 替换优化器（如 `Adam`）只需改一行，其余流程不变。
- 学习率调度让训练前期快速下降、后期精细收敛。
- 多组参数可用不同 `lr`（如特征提取层小、分类头大）分组传入。

**4. 最终复盘**

- **易出问题**：
  1. 漏写 `zero_grad()`，梯度在多个 batch 间累加导致更新异常。
  2. 在 `backward()` 前未算 loss 或 loss 与计算图断开。
  3. 把 `scheduler.step()` 放在 epoch 外或每个 batch 误调用造成 lr 骤降。
- **解决方法**：
  1. 每次 `backward` 前固定调用 `optim.zero_grad()`。
  2. 确保 `loss` 由带梯度的张量计算得到（未进 `torch.no_grad`）。
  3. 按调度器语义把 `scheduler.step()` 放在 epoch 末尾（或 iter 末尾）。
- **踩坑原因**：根因是 PyTorch 梯度默认累积而非覆盖，且优化器只读取已存在于叶子参数的 `.grad`，任一步缺失都会让参数更新基于错误或陈旧的梯度。
- **更优解**：
  1. 用 `torch.optim.Adam`/`AdamW` 通常收敛更快更稳。
  2. 用 `amp` 混合精度 + `optimizer.step()` 加速大模型训练。

## 3.9 dataset.py / dataloader.py 数据加载

**1. 初步讲解**

- `Dataset` 定义“单个样本如何读取”，需实现 `__getitem__` 与 `__len__`。
- 自定义 `Dataset` 可从文件夹、CSV、数据库等任意来源取数据。
- `DataLoader` 负责批量打包、打乱、并行读取，输出 `(imgs, targets)` 批次。
- 二者解耦：换数据源只改 `Dataset`，换读取方式只改 `DataLoader`。

**2. 核心代码**

```python
# dataset.py —— 自定义数据集
class mydata(Dataset):
    def __init__(self, root_dir, label_dir):
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_item_path = os.path.join(self.root_dir, self.label_dir, self.img_path[idx])
        img = Image.open(img_item_path)
        label = self.label_dir        # 用文件夹名当标签
        return img, label

    def __len__(self):
        return len(self.img_path)

train_dataset = mydata("dataself/train", "ants_image") + mydata("dataself/train", "bees_image")

# dataloader.py —— 批量加载
test_data = torchvision.datasets.CIFAR10('dataset', train=False,
                                          transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True,
                          num_workers=0, drop_last=False)
for data in test_loader:
    imgs, targets = data             # imgs: (64,3,32,32)
```

**3. 落地场景**

- 自己的图片数据按 `dataset.py` 写法组织成可训练数据集。
- `DataLoader` 的 `batch_size` 决定一次喂入多少样本，`shuffle=True` 防过拟合。
- 多个 `Dataset` 可用 `+` 拼接（源码中蚂蚁+蜜蜂），扩充训练集。
- 配合 3.10 的 `transform` 在读取时统一做预处理。

**4. 最终复盘**

- **易出问题**：
  1. `__getitem__` 返回 PIL 图像未转 tensor，后续卷积层报错。
  2. `label` 用字符串文件夹名，与 `CrossEntropyLoss` 要求的整数索引不符。
  3. `num_workers>0` 在 Windows 下未加 `if __name__=='__main__'` 引发多进程报错。
- **解决方法**：
  1. 在 `Dataset` 或 `DataLoader` 外统一加 `ToTensor()` transform。
  2. 建立类别名到整数索引的映射字典再返回标签。
  3. Windows 把训练入口包进 `if __name__ == '__main__':` 或 `num_workers=0`。
- **踩坑原因**：根因是 `Dataset` 只负责取数、`DataLoader` 只负责打包，二者都不自动做类型/设备转换，返回格式不符合模型与损失函数预期就会在训练第一步崩溃。
- **更优解**：
  1. 用 `torchvision.datasets.ImageFolder` 直接按子文件夹生成带索引标签的数据集。
  2. 大文件用 `num_workers>0` + `pin_memory=True` 提升数据吞吐。

## 3.10 Transforms.py / Tensorboard.py 数据变换与可视化

**1. 初步讲解**

- `transforms` 是数据预处理流水线，常见 `ToTensor`（PIL/numpy→张量并归一化到 [0,1]）、`Resize`、`Compose` 串联多个变换。
- `SummaryWriter`（TensorBoard）用于训练过程可视化：标量曲线、图像、计算图。
- `add_scalar` 画损失/准确率曲线，`add_image` 看样本，`add_graph` 看网络结构。
- 变换与可视化帮助理解数据分布、监控训练、定位问题。

**2. 核心代码**

```python
# Transforms.py —— 图像转张量并写入 TensorBoard
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img = Image.open(r'dataself\train\ants_image\0013035.jpg')
writer = SummaryWriter('logs')

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)             # PIL -> (C,H,W) 张量, 值归一到[0,1]
writer.add_image('Tensor_img', tensor_img)
writer.close()

# Tensorboard.py —— 标量/图像/计算图
writer = SummaryWriter('logs')
for i in range(100):
    writer.add_scalar('y=3x', 3*i, i)      # tag, value, step
img_array = np.array(Image.open(img_path))
writer.add_image('test', img_array, 1, dataformats='HWC')  # numpy 需指定格式

# 计算图(model.py 中)
writer.add_graph(xiaofan, input)           # 把网络结构画进 TensorBoard
```

**3. 落地场景**

- 训练前用 `ToTensor` 统一把图片转成模型可吃的标准张量。
- 用 `Compose([Resize, ToTensor])` 在 3.15 推理时把任意尺寸图规整到 32x32。
- 训练循环里周期调用 `writer.add_scalar` 记录 `train_loss`/`test_accuracy`。
- 用 `add_image` 抽查增强后的样本，确认预处理正确。

**4. 最终复盘**

- **易出问题**：
  1. `ToTensor` 前未 `convert('RGB')`，RGBA/灰度图通道数不一致。
  2. numpy 数组 `add_image` 不指定 `dataformats='HWC'` 导致图像错位。
  3. TensorBoard 日志路径混乱，多个实验曲线叠在一起难区分。
- **解决方法**：
  1. 读取后统一 `image.convert('RGB')` 再 `ToTensor`。
  2. numpy 图像显式传 `dataformats='HWC'`（或转成 `(C,H,W)` 张量）。
  3. 每次实验用独立 `logs/实验名` 目录，启动 `tensorboard --logdir=logs`。
- **踩坑原因**：根因是 TensorBoard 与 transforms 都强依赖张量布局约定（通道顺序、值域），一旦 PIL/numpy/张量的格式假设错位就会渲染异常或数值越界。
- **更优解**：
  1. 用 `torchvision.transforms.v2` 获得更强的数据增强与类型安全。
  2. 配合 `torch.utils.tensorboard` 的 `add_histogram` 监控权重分布。

## 3.11 model_pretrained.py 预训练模型与迁移学习

**1. 初步讲解**

- `torchvision.models` 提供在 ImageNet 上预训练好的成熟网络（如 VGG16）。
- `pretrained=True` 加载含权重，`False` 只加载结构（随机初始化）。
- 迁移学习：复用预训练特征提取器，只修改/新增最后分类层适配新任务。
- 两种改法：`.classifier.add_module('add_linear', ...)` 追加层，或 `classifier[6] = Linear(...)` 替换层。

**2. 核心代码**

```python
import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true  = torchvision.models.vgg16(pretrained=True)

train_data = torchvision.datasets.CIFAR10('dataset', train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)

# 方式一: 在 classifier 末尾追加一层, 1000类 -> 10类
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))

# 方式二: 直接替换第6个子层为新的线性层
vgg16_false.classifier[6] = nn.Linear(4096, 10)
```

**3. 落地场景**

- 小数据集上直接微调 VGG/ResNet，比从零训练快且精度高。
- 只训练新增的分类头、冻结前面卷积层，可大幅减少算力。
- 把预训练 backbone 当作 3.6 网络的替代特征提取器。
- 部署时只需保存改动部分，降低模型体积。

**4. 最终复盘**

- **易出问题**：
  1. 预训练输入尺寸/归一化与自己的数据不一致，特征分布错位。
  2. 替换层索引 `classifier[6]` 写错，覆盖到非预期子层。
  3. 忘了冻结原有权重，小数据上把预训练特征冲掉（灾难性遗忘）。
- **解决方法**：
  1. 复用原模型的 `transforms` 归一化均值方差预处理输入。
  2. 先 `print(model.classifier)` 确认层序号再替换。
  3. 训练初期 `requires_grad=False` 冻结 backbone，只训新头。
- **踩坑原因**：根因是预训练权重绑定了特定的输入分布与层结构，输入规范或层索引偏差会让特征提取失效，未加冻结则小数据噪声会破坏已学通用特征。
- **更优解**：
  1. 用 `torchvision.models` 的 `weights=` 新接口取代废弃的 `pretrained`。
  2. 采用分层学习率：backbone 小 lr、分类头大 lr 做精细微调。

## 3.12 model_save.py / model_load.py 模型保存与加载

**1. 初步讲解**

- 方式一：`torch.save(model, path)` 保存“结构+参数”整个对象（用 `pickle`）。
- 方式二：`torch.save(model.state_dict(), path)` 只保存参数字典，更轻量通用。
- 加载方式一：`torch.load(path)` 直接得模型，但需原类定义在环境中。
- 加载方式二：先建结构再 `load_state_dict(torch.load(path))`，推荐跨环境使用。
- `map_location` 可在无 GPU 时把权重映射到 CPU。

**2. 核心代码**

```python
# model_save.py —— 两种保存
import torch, torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
torch.save(vgg16, "vgg16_method1.pth")            # 方式一: 结构+参数
torch.save(vgg16.state_dict(), "vgg16_method2.pth") # 方式二: 仅参数

# model_load.py —— 两种加载
model = torch.load("vgg16_method1.pth")           # 方式一加载

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))  # 方式二加载

# 跨设备加载(3.15 推理示例)
model = torch.load("code-py/tudui_29_gpu.pth", map_location=torch.device('cpu'))
```

**3. 落地场景**

- 训练结束（3.13）按 epoch 存检查点，便于断点续训与选最优。
- 部署时常用方式二，只带参数文件更小、不依赖原代码路径。
- 推理脚本 `test.py` 用 `map_location='cpu'` 在没显卡的机器上加载 GPU 模型。
- 保存检查点建议同时存 `state_dict` 与优化器状态，方便恢复训练。

**4. 最终复盘**

- **易出问题**：
  1. 方式一加载时原类（如 `xiaofan`）不在当前作用域，报 `AttributeError`。
  2. 方式二加载前未先实例化网络结构，无处接收参数。
  3. GPU 训的模型直接 `torch.load` 到无 CUDA 环境报设备错误。
- **解决方法**：
  1. 加载方式一前 `from model import *` 导入类定义，或改用方式二。
  2. 方式二严格先建结构再 `load_state_dict`。
  3. 跨设备加 `map_location=torch.device('cpu')`。
- **踩坑原因**：根因是方式一用 pickle 序列化整个对象、强依赖原类代码，方式二只存张量字典、需调用方先重建结构，设备不一致时还需显式重映射张量。
- **更优解**：
  1. 统一用方式二 + 单独保存网络定义，迁移性最强。
  2. 用 `torch.save({'model': sd, 'optim': sd, 'epoch': e}, ckpt)` 存完整检查点。

## 3.13 train.py / train_cpu.py 完整训练流程

**1. 初步讲解**

- 这是把前面所有模块串起来的核心训练脚本：数据→网络→损失→优化→保存。
- 流程：准备数据集 → DataLoader → 建模型 → 设损失/优化器 → 多轮 train/test → 存模型。
- 训练阶段 `model.train()` 开启 dropout/batchnorm 训练行为，测试阶段 `model.eval()` 关闭。
- 用 `torch.no_grad()` 包裹测试，省显存、关梯度加速推理。

**2. 核心代码**

```python
# train_cpu.py / train.py —— 完整训练（最关键）
from model import *                       # 引入 3.6 的网络
train_data = torchvision.datasets.CIFAR10(root="dataset", train=True,
                                           transform=torchvision.transforms.ToTensor(), download=True)
test_data  = torchvision.datasets.CIFAR10(root="dataset", train=False,
                                           transform=torchvision.transforms.ToTensor(), download=True)
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader  = DataLoader(test_data,  batch_size=64)

xiaofan = xiaofan()                       # 网络
loss_fn = nn.CrossEntropyLoss()           # 损失
optimizer = torch.optim.SGD(xiaofan.parameters(), lr=1e-2)  # 优化器
epoch = 10
writer = SummaryWriter("logs_train")

for i in range(epoch):
    # ---- 训练 ----
    xiaofan.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = xiaofan(imgs)           # 前向
        loss = loss_fn(outputs, targets)  # 算损失
        optimizer.zero_grad()             # 清梯度
        loss.backward()                   # 反向
        optimizer.step()                  # 更新
        total_train_step += 1
        if total_train_step % 100 == 0:
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # ---- 测试 ----
    xiaofan.eval()
    total_test_loss, total_accuracy = 0, 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = xiaofan(imgs)
            total_test_loss += loss_fn(outputs, targets).item()
            total_accuracy += (outputs.argmax(1) == targets).sum()
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    torch.save(xiaofan, "xiaofan_{}.pth".format(i))   # 每轮存模型
writer.close()
```

**3. 落地场景**

- 任何监督学习任务都套用此模板，仅替换数据集、网络、超参即可。
- `train.py` 与 `train_cpu.py` 区别仅在于是否把模型/数据搬上 GPU（见 3.14）。
- 通过 `writer` 实时看 loss 下降、accuracy 上升，判断训练是否健康。
- 按 `total_test_step` 记录最佳轮次，挑 `test_accuracy` 最高的 `.pth` 部署。

**4. 最终复盘**

- **易出问题**：
  1. 训练前忘记 `model.train()`、测试前忘记 `model.eval()`，batchnorm/dropout 行为错。
  2. 测试阶段没包 `torch.no_grad()`，白白占用显存且拖慢。
  3. loss 直接 `print(loss)` 而非 `loss.item()`，计算图不释放导致显存泄漏。
  4. 每轮全量测试却没 `zero_grad`，无关但易混淆；或保存路径写死覆盖。
- **解决方法**：
  1. 训练循环开头 `model.train()`，测试循环开头 `model.eval()` 成对写。
  2. 测试/验证统一用 `with torch.no_grad():` 包裹。
  3. 记录标量用 `loss.item()` 取 Python 数值，断开计算图。
  4. 保存文件名带入 epoch 索引，避免互相覆盖。
- **踩坑原因**：根因是训练与测试阶段模型内部状态（梯度、归一化统计量、dropout 掩码）不同，`train()/eval()` 与 `no_grad()` 控制这些状态，缺失会让指标失真或资源浪费。
- **更优解**：
  1. 用 `torchmetrics` 或早停（EarlyStopping）自动挑最优 checkpoint。
  2. 用 `tqdm` 包装 dataloader 显示进度，用 AMP 自动混合精度加速。

## 3.14 train_gpu_1.py / train_gpu_2.py GPU 加速

**1. 初步讲解**

- GPU 用并行核心加速矩阵运算，深度学习训练默认优先使用 CUDA。
- `train_gpu_1.py`：用 `if torch.cuda.is_available()` 判断后 `.cuda()` 把模型/数据搬上显卡。
- `train_gpu_2.py`：更规范地用 `device = torch.device("cuda")` + `.to(device)` 统一管理设备。
- 模型、损失函数、每个 batch 的输入与目标都必须处于同一设备，否则报错。

**2. 核心代码**

```python
# train_gpu_1.py —— 判断式 .cuda()
xiaofan_model = xiaofan()
if torch.cuda.is_available():
    xiaofan_model = xiaofan_model.cuda()
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 每个 batch:
if torch.cuda.is_available():
    imgs = imgs.cuda(); targets = targets.cuda()

# train_gpu_2.py —— 统一 device 写法（推荐）
device = torch.device("cuda")
xiaofan_model = xiaofan().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
# 每个 batch:
imgs = imgs.to(device); targets = targets.to(device)
```

**3. 落地场景**

- 把 3.13 的 CPU 训练脚本升级为 GPU 训练，速度可提升数倍到数十倍。
- 多卡环境可进一步 `torch.nn.DataParallel` 或 `DistributedDataParallel`。
- 推理脚本 `test.py` 反向用 `map_location='cpu'` 在没显卡环境加载 GPU 模型。
- 调试时先小数据在 CPU 跑通，再切 GPU 避免设备相关报错干扰逻辑。

**4. 最终复盘**

- **易出问题**：
  1. 只把模型 `.cuda()` 却忘把输入 `imgs/targets` 也搬上 GPU，报 device 不匹配。
  2. 损失函数没搬设备，标签在 CPU、输出在 GPU 冲突。
  3. `.cuda()` 硬编码，环境无显卡时直接崩溃。
- **解决方法**：
  1. 统一用 `device` 变量，模型、损失、每批数据都 `.to(device)`。
  2. 封装 `to(device)` 工具函数，避免遗漏。
  3. 用 `torch.device("cuda" if torch.cuda.is_available() else "cpu")` 兼容两种环境。
- **踩坑原因**：根因是 PyTorch 张量运算要求参与计算的双方位于同一设备，模型与数据任何一侧未迁移都会触发跨设备访存错误，硬编码 `.cuda()` 则失去环境可移植性。
- **更优解**：
  1. 用 `device` 统一管理，整套代码零改动即可 CPU/GPU 切换。
  2. 大模型用 `DataParallel`/`DDP` 做多卡并行训练。

## 3.15 test.py 模型推理测试

**1. 初步讲解**

- 推理阶段用训练好的模型对单张/批量图片做前向预测，得到类别。
- 流程：加载图片 → `convert('RGB')` → `transforms` 规整尺寸 → 加载模型 → `eval()`+`no_grad` → `argmax` 取预测类。
- `model.eval()` 关闭 dropout/batchnorm 训练态，`torch.no_grad()` 关梯度省资源。
- `output.argmax(1)` 取每行最大 logit 的索引，即预测类别。

**2. 核心代码**

```python
from PIL import Image
import torch, torchvision
from torch import nn

image = Image.open("imgs/airplane.png").convert('RGB')   # 统一转 RGB
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
image = transform(image)                                  # (3,32,32)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(                       # 同 3.6 结构
            nn.Conv2d(3, 32, 5, 1, 2), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*4*4, 64), nn.Linear(64, 10))

    def forward(self, x):
        return self.model(x)

# 加载 GPU 训练的模型到 CPU
model = torch.load("code-py/tudui_29_gpu.pth", map_location=torch.device('cpu'))
image = torch.reshape(image, (1, 3, 32, 32))             # 补 batch 维
model.eval()
with torch.no_grad():
    output = model(image)
print(output.argmax(1))                                   # 预测类别索引
```

**3. 落地场景**

- 训练完成后对真实业务图片做预测，验证模型可用性。
- 与 3.12 配合：加载检查点 → 预处理 → 推理，形成部署闭环。
- 单图推理必须补 `(1, C, H, W)` 的 batch 维，否则卷积层报错。
- 可作为 Web/API 服务的前置逻辑，把图片转张量送模型得结果。

**4. 最终复盘**

- **易出问题**：
  1. 推理前没 `model.eval()`，batchnorm 用 batch 统计量导致结果漂移。
  2. 忘记补 batch 维，单图 `(3,32,32)` 直接进卷积层维度不符。
  3. 没 `convert('RGB')`，PNG 带透明通道变 4 通道与 3 通道卷积不匹配。
  4. 加载 GPU 模型到 CPU 却没 `map_location`，报设备错误。
- **解决方法**：
  1. 推理固定 `model.eval()` 包裹在 `torch.no_grad()` 中。
  2. 输入用 `image.unsqueeze(0)` 或 `reshape(1, ...)` 补 batch 维。
  3. 读取后统一 `.convert('RGB')` 再转 tensor。
  4. 跨设备加载显式 `map_location=torch.device('cpu')`。
- **踩坑原因**：根因是推理与训练在模型状态、输入维度、通道数、设备四个方面都有约定，任一处与训练时不一致都会让前向传播失败或预测失真。
- **更优解**：
  1. 把预处理封装成与训练一致的 `transforms`，保证分布统一。
  2. 用 `torch.jit.script`/`trace` 导出模型，脱离 Python 类定义做高性能部署。

---

# 第四板块： 场景应用

## 4.1 实战场景

**用本套知识做一个「宠物（猫/狗）图像分类器」并接入业务系统**。下面是每个知识点在场景里的角色与原因：

| 知识点                      | 场景中的作用           | 这样做的原因                   |
| ------------------------ | ---------------- | ------------------------ |
| `nn.Module`/`Sequential` | 定义分类网络骨架         | 没有结构模型就不存在，参数无法登记        |
| `Conv2d`                 | 从猫狗图提取耳朵/毛色等局部特征 | 卷积保空间结构、共享权重，远优于全连接      |
| `MaxPool2d`              | 压缩特征、扩大感受野       | 降低计算量、提升对姿态变化的鲁棒性        |
| `ReLU`                   | 加入非线性            | 否则网络表达力仅限线性，分不清复杂边界      |
| `Linear`                 | 输出"猫/狗"两类分数      | 把高维特征映射成最终决策             |
| `CrossEntropyLoss`       | 量化预测与标签差距        | 给优化器一个明确的下降目标            |
| `SGD`+`backward`         | 迭代更新权重           | 让模型从随机初始化逐步变准            |
| `Dataset`/`DataLoader`   | 加载你的猫狗照片         | 把硬盘图片变成 batch 张量喂给模型     |
| `transforms.ToTensor`    | 图片转张量并归一化        | 模型只接受定格式张量               |
| `TensorBoard`            | 看 loss/准确率曲线     | 及时发现过拟合/不收敛，决定调参         |
| `model_pretrained`       | 用 ResNet 预训练权重   | 数据少也能高精度，避免从头训           |
| `model_save/load`        | 训练成果固化、部署        | 训练一次，多次推理，模型可发布          |
| `train_cpu`              | 端到端训练出可用模型       | 上述所有环节缺一则无法产出模型          |
| `train_gpu`              | 用显卡加速训练          | 数据量大时 CPU 训练慢数十倍         |
| `test`                   | 上线后对新照片实时预测      | `eval()+no_grad` 保证推理快且稳 |

## 4.2 经验总结

- **卷积/池化/激活 = 让模型"看懂图"；线性/损失/优化 = 让模型"会判断并进步"；数据/变换 = 让模型"吃得到数据"；保存/加载/GPU/推理 = 让模型"能用、好用、上线"。**
- **工程上记住三处保命：`zero_grad()` 必清、`eval()/no_grad()` 测试必加、`to(device)` 统一设备管理**。

## 4.3 后续拓展

- 换更大 backbone（ResNet50）+ 预训练 → 精度再升。
- 加 `transforms.RandomCrop/Flip` 数据增强 → 提泛化。
- 用 `torch.jit` 或 `onnx` 导出 → 嵌入手机/边缘设备。
- 接 FastAPI → 封装成 HTTP 图像分类服务。
