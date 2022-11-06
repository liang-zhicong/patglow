import torch
from PIL import Image
import os

class glow_dataset(torch.utils.data.Dataset):
    def __init__(self,num_classes,transform,test=False,rotation_data=False):
        self.transform = transform
        self.rotation_data = rotation_data
        if not self.rotation_data:
            if test:
                self.root = "D:/lzc/64/test/" + str(num_classes)

            else:
                self.root = "D:/lzc/64/train/"+str(3)
                # 文件夹中3包含5万张图片训练
            self.paths = os.listdir(self.root)
            self.label = num_classes
        else:
            self.paths = []
            self.labels = []
            if test:
                root = "./dataset/glow-rotation/cifar10_rotation_glow_one_model/test/"
            else:
                root = "./dataset/glow-rotation/cifar10_rotation_glow_one_model/train/"
            for i in range(10):
                path_i = os.listdir(root+str(i))
                for path in path_i:
                    self.paths.append(root+str(i)+"/"+path)
                    self.labels.append(i)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,index):
        if not self.rotation_data:
            x = Image.open(os.path.join(self.root,self.paths[index]))
            x = self.transform(x)
            # filename=self.paths[index]
            label = self.label
        else:
            #print(self.paths[index])
            x = self.transform(Image.open(self.paths[index]))
            # filename=self.paths[index]
            label = self.labels[index]
        return x , label #, filename