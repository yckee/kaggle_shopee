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

from shopeenet import ShopeeNet, ShopeeBert, ShopeeCombined
from shopeedataset import ShopeeDataset, ShopeeTextDataset, ShopeeCombinedDataset

from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.neighbors import NearestNeighbors

IMG_SIZE = 256
N_WORKERS = 4
BATCH_SIZE = 8
EPOCHS = 12
INIT_LR = 3e-4
MIN_LR = 1e-6
SEED = 24
FOLD_ID = 0

# MODEL_PARAMS = {
#     'model_name' : 'eca_nfnet_l0', # 'effecientnet_b0'
#     'feature_space' : 512, 
#     'out_features' : 11014, 
#     'scale' : 12.0, 
#     'margin' : 0.6
# }

MODEL_PARAMS = {
    'model_name' : 'distilbert-base-multilingual-cased', # 'effecientnet_b0'
    'feature_space' : 512, 
    'out_features' : 11014, 
    'scale' : 12.0, 
    'margin' : 0.35
}

COMBINED_MODEL_PARAMS = {
    'cnn_name' : 'eca_nfnet_l0',
    'bert_name' : 'distilbert-base-multilingual-cased', # 'effecientnet_b0'
    'feature_space' : 512, 
    'out_features' : 11014, 
    'scale' : 12.0, 
    'margin' : 0.35    
}


def getMetric(col):
    def f1score(row):
        n = len(np.intersect1d(row.target, row[col]))
        return 2 * n / (len(row.target) + len(row[col]))
    return f1score

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch(SEED)

    #---------------------------------------------Model---------------------------------------------
    # model = ShopeeNet(**MODEL_PARAMS)
    # model.to('cuda')

    # model = ShopeeBert(**MODEL_PARAMS)
    # model.to('cuda')

    model = ShopeeCombined(**COMBINED_MODEL_PARAMS)
    model.to('cuda')

    #-------------------------------------Data loading & preproccessing----------------------------
    path = 'd:/projects/kaggle/shopee/data/'
    
    df = pd.read_csv(path + 'folds.csv')
    # df = pd.read_csv(path + 'train.csv')
    
    tmp = df.groupby('label_group').posting_id.unique().to_dict()
    df['target'] = df.label_group.map(tmp)

    le = LabelEncoder()
    df.label_group = le.fit_transform(df.label_group)

    # df_train = df
    
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

    # dataset_train = ShopeeDataset(df_train, path, 'train',  transform = transforms_train)
    # dataset_valid = ShopeeDataset(df_valid, path, 'train',  transform = transforms_valid)

    # dataset_train = ShopeeTextDataset(df_train, 'train', tokenizer=model.tokenizer)
    # dataset_valid = ShopeeTextDataset(df_valid, 'train', tokenizer=model.tokenizer)

    dataset_train = ShopeeCombinedDataset(df_train, path, 'train', transform = transforms_train, tokenizer=model.tokenizer)
    dataset_valid = ShopeeCombinedDataset(df_valid, path, 'train', transform = transforms_valid, tokenizer=model.tokenizer)    


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
   



    #-------------------------------------Loss function, optimizer, sheduler------------------------
    criterion = nn.CrossEntropyLoss()
    criterion.to('cuda')
    optimizer = optim.Adam(model.parameters(), lr = INIT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=MIN_LR)

    #---------------------------------------------Train---------------------------------------------
    best_loss = 100_000
    best_epoch = -1
    os.makedirs(f"checkpoints/{model.saving_name}", exist_ok=True)

    for epoch in range(EPOCHS):
        loss_train = train(loader_train, model, criterion, optimizer, scheduler, epoch)
        loss_valid = eval(loader_valid, model, criterion, epoch, df_valid)

        # if loss_valid < best_loss:
        #     best_loss = loss_valid
        #     best_epoch = epoch
        torch.save(model.state_dict(), f"checkpoints/{model.saving_name}/{model.saving_name}_epoch{epoch}.pth")

    # print(f"Best val  score: {best_loss} epoch: {best_epoch}")

def train(loader, model, criterion, optimizer, sheduler, epoch):
    model.train()
    bar = tqdm(loader)
    losses = []

    for images, input_ids, attention_mask, targets in bar:
        
        # images, targets = images.to('cuda'), targets.to('cuda').long()   SHopeeNEt
       
       
        # input_ids = images[0].to('cuda')      SHopeeBert
        # attention_mask = images[0].to('cuda')
        # targets = targets.to('cuda').long()     

        images, input_ids, attention_mask, targets = images.to('cuda'), input_ids.to('cuda'), attention_mask.to('cuda'), targets.to('cuda').long()

        logits = model(images, input_ids, attention_mask, targets)
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


def eval(loader, model, criterion, epoch, df):
    model.eval()
    bar = tqdm(loader)

    losses = []
    # img_embeds = []

    with torch.no_grad():
        for images, input_ids, attention_mask, targets in bar:

            images, input_ids, attention_mask, targets = images.to('cuda'), input_ids.to('cuda'), attention_mask.to('cuda'), targets.to('cuda').long()
    

            logits = model(images, input_ids, attention_mask, targets)
            loss = criterion(logits, targets)

            # images, targets = images.to('cuda'), targets.to('cuda').long()      

            # logits = model(images, targets)
            # loss = criterion(logits, targets)

            # logits = logits.reshape(logits.shape[0], logits.shape[1])
            # emb = logits.detach().cpu().numpy()
            # emb = normalize(emb, copy=False).astype(dtype=np.float16)
            # img_embeds.append(emb)
            
            losses.append(loss.item())

            bar.set_description(f'Eval Loss: {np.mean(losses):.5f}, Epoch: {epoch}')

    # img_embeds = np.concatenate(img_embeds)
    # best_th = get_best_image_th(df, img_embeds)

    loss_eval = np.mean(losses)
    return loss_eval

def get_best_image_th(df, embeddings):
    KNN = 50
    model = NearestNeighbors(n_neighbors = KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    
    thresholds = list(np.arange(0.6, 0.9, 0.02))
    bar = tqdm(thresholds)
    scores = []
    
    for th in bar:
        preds = []
        for dist, idx in zip(distances, indices):
            posting_ids = df.iloc[np.asnumpy(idx[dist < th])].posting_id.values
            preds.append(posting_ids)
        
        df['tmp'] = preds
        df['f1'] = df.apply(getMetric('tmp'), axis=1)
        score = df.f1.mean()
        scores.append(score)
        bar.set_description(f"Threshold: {th:.4f} - Score: {score:.4f} Best threshold: {thresholds[np.argmax(scores)]:.4f} - Score: {max(scores):.4f}")
    
    return thresholds[np.argmax(scores)]    


if __name__ == '__main__':
    main()


