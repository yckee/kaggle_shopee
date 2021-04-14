import pandas as pd

import cv2
import torch
from torch.utils.data.dataset import Dataset

class ShopeeDataset(Dataset):
    def __init__(self, df, mode, transform):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = cv2.imread(row.image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']

        if self.mode == 'test':
            return img
        if self.mode == 'train':
            return img, torch.tensor(row.label_group).float()
    
    def __len__(self):
        return len(self.df)