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


class ShopeeTextDataset(Dataset):
    def __init__(self, df, mode, tokenizer):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.tokenizer = tokenizer

        
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        text = row.title

        text = self.tokenizer(text, max_length=128, padding='max_length', truncation=True,  return_tensors='pt')
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0] 

        if self.mode == 'test':
            return (input_ids, attention_mask)
        if self.mode == 'train':
            return (input_ids, attention_mask), torch.tensor(row.label_group).float()
    
    def __len__(self):
        return len(self.df)




class ShopeeCombinedDataset(Dataset):
    def __init__(self, df, path,  mode, transform, tokenizer):
        self.df = df.reset_index(drop=True)
        self.path = path
        self.mode = mode
        self.transform = transform
        self.tokenizer = tokenizer

        
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        text = row.title

        text = self.tokenizer(text, max_length=128, padding='max_length', truncation=True,  return_tensors='pt')
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0] 

        img = cv2.imread(self.path + 'train_images/' + row.image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']

        if self.mode == 'test':
            return img, input_ids, attention_mask
        if self.mode == 'train':
            return img, input_ids, attention_mask, torch.tensor(row.label_group).float()
    
    def __len__(self):
        return len(self.df)