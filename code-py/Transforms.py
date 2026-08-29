# %%
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# img_path = r'D:\wyf\study\python\pytorch-test\data\train\ants_image\0013035.jpg'
img_path = r'dataself\train\ants_image\0013035.jpg'
img = Image.open(img_path)

writer = SummaryWriter('logs')

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image('Tensor_img',tensor_img)
writer.close()
# %%
