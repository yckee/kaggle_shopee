import os, random
import pandas as pd 
import numpy as np
from tqdm import tqdm

import torch
torch.cuda.empty_cache()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.dataloader import DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from shopeenet import ShopeeNet
from shopeedataset import ShopeeDataset

from sklearn.preprocessing import LabelEncoder

IMG_SIZE = 512
N_WORKERS = 4
BATCH_SIZE = 8
EPOCHS = 30
INIT_LR = 3e-4
MIN_LR = 1e-6
SEED = 24
FOLD_ID = 4

MODEL_PARAMS = {
    'feature_space' : 512, 
    'out_features' : 11014, 
    'scale' : 64, 
    'margin' : 0.5
}


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch(SEED)
    #-------------------------------------Data loading & preproccessing----------------------------
    path = 'd:/projects/kaggle/shopee/data/'
    
    df = pd.read_csv(path + 'folds.csv')

    le = LabelEncoder()
    df.label_group = le.fit_transform(df.label_group)

    df_train = df[df['fold'] != FOLD_ID]
    df_valid = df[df['fold'] == FOLD_ID]

    transforms_train = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
        A.HueSaturationValue(p=0.5, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
        A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
        A.CoarseDropout(p=0.5),
        A.Normalize(),
        ToTensorV2()

    ])

    transforms_valid = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])

    dataset_train = ShopeeDataset(df_train, 'train',  transform = transforms_train)
    dataset_valid = ShopeeDataset(df_valid, 'train',  transform = transforms_valid)

    loader_train = DataLoader(
        dataset_train, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True,
        drop_last=True, 
        num_workers=N_WORKERS
    )
    loader_valid = DataLoader(
        dataset_valid, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=N_WORKERS
    )
   
    #---------------------------------------------Model---------------------------------------------
    model = ShopeeNet(**MODEL_PARAMS)
    model.to('cuda')

    #-------------------------------------Loss function, optimizer, sheduler------------------------
    criterion = nn.CrossEntropyLoss()
    criterion.to('cuda')
    optimizer = optim.Adam(model.parameters(), lr = INIT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=MIN_LR)

    #---------------------------------------------Train---------------------------------------------
    best_loss = 100_000
    for epoch in range(EPOCHS):
        loss_train = train(loader_train, model, criterion, optimizer, scheduler, epoch)
        loss_valid = eval(loader_valid, model, criterion, epoch)

        if loss_valid < best_loss:
            best_loss = loss_valid
            torch.save(model.state_dict(), f"checkpoints/arcface_epoch{epoch}.pth")


def train(loader, model, criterion, optimizer, sheduler, epoch):
    model.train()
    bar = tqdm(loader)
    losses = []

    for images, targets in bar:
        images, targets = images.to('cuda'), targets.to('cuda').long()       

        logits = model(images, targets)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        bar.set_description(f'Train Loss: {np.mean(losses):.5f}, Epoch: {epoch}')

    if sheduler is not None:
        sheduler.step()

    loss_train = np.mean(losses)
    return loss_train


def eval(loader, model, criterion, epoch):
    model.eval()
    bar = tqdm(loader)

    losses = []

    with torch.no_grad():
        for images, targets in bar:
            images, targets = images.to('cuda'), targets.to('cuda').long()      

            logits = model(images, targets)
            loss = criterion(logits, targets)
            
            losses.append(loss.item())

            bar.set_description(f'Eval Loss: {np.mean(losses):.5f}, Epoch: {epoch}')

    loss_eval = np.mean(losses)
    return loss_eval

if __name__ == '__main__':
    main()


