from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter('logs')

for i in range(100):
    writer.add_scalar('y=3x',3*i,i)

img_path = r'dataself\train\bees_image\17209602_fe5a5a746f.jpg'   
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

writer.add_image('test',img_array,1,dataformats='HWC')
writer.close()

# %%
print(type(img_array))
print(img_array.shape)
# %%
