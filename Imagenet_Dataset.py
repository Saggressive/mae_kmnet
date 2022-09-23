from torch.utils.data import Dataset, DataLoader
import csv
from PIL import Image
from torchvision import transforms
# from numpy import random
import random
import os
import numpy as np
class Imagenet_Dataset(Dataset):
    def __init__(self, path, input_size):
        self.image_path = []
        images_folder_path=os.listdir(path)
        for folder in images_folder_path:
            folder_path=path+os.sep+folder
            folder_images_list=os.listdir(folder_path)
            images_path=[folder_path+os.sep+i for i in folder_images_list]
            self.image_path.extend(images_path)

        self.transform_base = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip()])
        self.transform_target = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.1, 0.05)], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mif = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        path = self.image_path[item]
        image = Image.open(path).convert("RGB")
        base_image = self.transform_base(image)
        pixel_shift = random.randint(0,31)
        mif_image = base_image.crop((0,0,224,224))
        tatget_image = base_image.crop((pixel_shift,pixel_shift,224+pixel_shift,224+pixel_shift))
        mif_image = self.transform_mif(mif_image)
        tatget_image = self.transform_target(tatget_image)
        return mif_image , tatget_image
