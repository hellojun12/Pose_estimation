import os
from glob import glob
import numpy as np
import json
import cv2
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.wandb_img import show_img_wandb
from utils.average_meter import AverageMeter

import datasets
import models

from losses import HeatmapMSELoss
from torch.optim import Adam

import wandb
model_names = sorted(name for name in models.__dict__)
dataset_names = sorted(name for name in models.__dict__)

def train(args):

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    cuda = torch.device('cuda')
   
    print(f"Creating model.....")
    model = models.__dict__[args.architecture](num_stacks=args.num_stacks,
                                               num_blocks=args.num_blocks,
                                               num_classes=args.num_classes).cuda()
  
    print(f"Creating training dataset...")
    train_dataset = datasets.__dict__[args.dataset](is_train=True, **vars(args))
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              num_workers=args.workers, 
                              pin_memory=True)


    print(f"Creating valid dataset...")
    val_dataset = datasets.__dict__[args.dataset](is_train=False, **vars(args))
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers, 
                            pin_memory=True)

    print(f"train_loader length: {len(train_loader)}")
    print(f"valid_loader length: {len(val_loader)}")
    criterion = HeatmapMSELoss(False)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
    
        train_loss, valid_loss = AverageMeter(), AverageMeter()

        # Training
        model.train()
        for iter, (img, hm_gt) in enumerate(train_loader):

            img, hm_gt = img.to(dtype=torch.float32, device=cuda), hm_gt.to(device=cuda)
       
            pred_logit = model(img)
            
            loss = 0
            for pred in pred_logit:

                loss += criterion(pred, hm_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), len(img))

            print("\rEpoch [%3d/%3d] | Iter [%3d/%3d] | Train Loss %.4f" % (epoch+1, epochs, iter+1, len(train_loader), train_loss.avg), end='')
            wandb.log({"Train loss": train_loss.avg})

        # Validation
        model.eval()
        for iter, (img, hm_gt) in enumerate(val_loader):
            img, hm_gt = img.to(dtype=torch.float32, device=cuda), hm_gt.to(device=cuda)

            with torch.no_grad():
                pred_logit = model(img)
        
            loss = 0
            for pred in pred_logit:

                loss += criterion(pred, hm_gt)

            valid_loss.update(loss.item(), len(img))
    
        print("\nEpoch [%3d/%3d] | Iter [%3d/%3d] | Valid Loss %.4f" % (epoch+1, epochs, iter, len(val_loader), valid_loss.avg))
        wandb.log({"Valid loss": valid_loss.avg})

        wandb.log({"Table": show_img_wandb(img, hm_gt, pred_logit)})

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='lsp')
    parser.add_argument("--model", type=str)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--resize", type=str, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs",  type=int ,default=100)
    parser.add_argument("--lr", type=int, default=1e-5)
    parser.add_argument("--img_resize", type=list)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--criterion", type=str, default="MSE")
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--resnet_layers", default=50, type=int, metavar='N')
    parser.add_argument("--architecture", "-arch", default="hg")
    parser.add_argument("--num_blocks", default=1, type=int)
    parser.add_argument("--num_stacks", default=2, type=int)
    parser.add_argument("--num_classes", default=14, type=int)

    parser.add_argument("--input_size", default=32, type=int)
    parser.add_argument("--output_size", default=32, type=int)
    parser.add_argument("--n_landmarks", default=14, type=int)
    parser.add_argument("--sigma", default=1.5, type=int)

    args = parser.parse_args()

    wandb.login()
    wandb_run = wandb.init(project='Pose_estimation', entity='bc11')
    wandb.config = {
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size
            }

    train(args)
    

    
