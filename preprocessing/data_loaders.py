import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomData(Dataset):
    """ dataset class to load images from directory """
    def __init__(self, path):
        self.path = path
        # get classes from folder names
        self.classes = sorted(os.listdir(path))
        self.data = []
        
        # save all img paths and their labels
        for y, c in enumerate(self.classes):
            c_path = os.path.join(path, c)
            if os.path.isdir(c_path):
                for img in os.listdir(c_path):
                    self.data.append((os.path.join(c_path, img), y))
                    
        # basic image transforms
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        """ return total number of samples """
        return len(self.data)

    def __getitem__(self, i):
        """ return one sample (image, label) """
        p, y = self.data[i]
        
        # open image in rgb mode
        x = Image.open(p).convert('RGB')
        x = self.tf(x)
        
        return x, y

def get_loader(path, bs=32):
    """ create and return dataloader """
    ds = CustomData(path)
    return DataLoader(ds, batch_size=bs, shuffle=True)