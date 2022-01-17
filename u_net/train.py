import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import u_net
from dataset import dataset
from utils import train_augment, val_augment
from torch.utils.data import DataLoader
import zipfile

learning_rate = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOUHS = 100
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
NUM_WORKERS = 2
TRAIN_IMAGE_PATH ='../output/kaggle/working/train/*.jpg'
TRAIN_MASK_PATH = '../output/kaggle/working/train_masks/*.gif'


def train(loader, model, optim, loss_function, scaler):
    for batch_index, (data, targets) in enumerate(loader):
        data = data.to(device = DEVICE)

        #adding a channel diminetion
        targets = targets.float().unsquuze(1).to(device = DEVICE)


        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_function(predictions, targets)

        # backward
        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()


def main():

    model = u_net(3, 1).to(DEVICE)
    loss_function = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    path_to_zip_file = "../input/carvana-image-masking-challenge/train.zip"
    directory_to_extract_to = "../output/kaggle/working"
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    path_to_zip_file = "../input/carvana-image-masking-challenge/train_masks.zip"
    directory_to_extract_to = "../output/kaggle/working"
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)



    train_dataset = dataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, transform = train_augment)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    # val_dataset = dataset(VAL_IMAGE_PATH, VAL_MASK_PATH, transform = train_augment)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOUHS):
        train(train_loader, model, optim, loss_function, scaler)



if __name__ == "__main__":
    main()


