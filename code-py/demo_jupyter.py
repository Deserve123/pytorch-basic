# %%
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("CUDA 不可用，请检查安装。")
# %%
import sys
print(sys.executable)
print("Hello, 交互式环境测试成功！")
# %%
import os
print(os.listdir("d:/wyf/study/python/pytorch-test/dataself/train/ants_image"))
#进行cwd当前工作目录的修改 固定在根目录而不是当前目录 避免在当前目录时而找不到与其平级目录下的文件
os.chdir(r'D:\wyf\study\python\pytorch-test') #例如 插件运行py时在code-py目录 无法找到data目录
print("当前工作目录:", os.getcwd())   
print(os.listdir("dataself/train/ants_image"))