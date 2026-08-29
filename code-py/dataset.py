# %%
from torch.utils.data import Dataset
from PIL import Image
import os

class mydata(Dataset):
    def __init__(self, root_dir, label_dir):  
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):               
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):                        
        return len(self.img_path)

root_dir = "dataself/train"
ants_label_dir = "ants_image"   
bees_label_dir = "bees_image"  

ants_dataset = mydata(root_dir, ants_label_dir)
bees_dataset = mydata(root_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset

print(f"蚂蚁图片数量: {len(ants_dataset)}")
print(f"蜜蜂图片数量: {len(bees_dataset)}")
print(f"训练集总数量: {len(train_dataset)}")

ants_dataset = mydata(root_dir, ants_label_dir)
img, label = ants_dataset[0]
img.show()
# %%
