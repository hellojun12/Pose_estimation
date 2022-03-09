import os
from glob import glob
import numpy as np
import json
import cv2
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.wandb_img import show_img_wandb
from utils.average_meter import AverageMeter
from utils.evaluation_tool import accuracy
from utils.darkpose import get_final_preds

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
    
    criterion = HeatmapMSELoss(False)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
    
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        val_losses = AverageMeter()
        val_acc = AverageMeter()

        # Training
        model.train()

        end = time.time()
        for iter, (img, hm_gt) in enumerate(train_loader):

            # measure data loading time
            data_time.update(time.time() - end)
            img, hm_gt = img.to(dtype=torch.float32, device=cuda), hm_gt.to(device=cuda)
       
            pred_logit = model(img)
            
            loss = 0

            if isinstance(pred_logit, list):

                loss = criterion(pred_logit[0], hm_gt) 

                for pred in pred_logit[1:]:
                    
                    loss += criterion(pred, hm_gt)
            
            else:
                pred = pred_logit
                loss = criterion(pred, hm_gt)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and recod loss
            losses.update(loss.item(), img.size(0))

            _, avg_acc, cnt, pred = accuracy(pred.detach().cpu().numpy(),
                                             hm_gt.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            #measure elapsed time
            batch_time.update(time.time() -end)
            end = time.time()
                                            

            print("\rEpoch [{0}][{1}/{2}] | Train_loss {loss.val:.5f} | Train_accuracy_value {acc.val:.5f}"
                      .format(epoch, iter, len(train_loader), loss=losses, acc=acc), end='')
                  
            wandb.log({"Epoch": epoch, "Train loss val": losses.val, 
                    "Train loss avg": losses.avg, "Accuracy val":acc.val, 
                    "Accuracy avg":acc.avg})
                    

        # Validation
        model.eval()
        for iter, (img, hm_gt) in enumerate(val_loader):
            img, hm_gt = img.to(dtype=torch.float32, device=cuda), hm_gt.to(device=cuda)

            with torch.no_grad():
                pred_logit = model(img)
        
            loss = 0
            if isinstance(pred_logit, list):

                loss = criterion(pred_logit[0], hm_gt) 

                for pred in pred_logit[1:]:
                   
                    loss += criterion(pred, hm_gt)
            
            else:
                pred = pred_logit
                loss = criterion(pred, hm_gt)

            val_losses.update(loss.item(), len(img))
            _, avg_acc, cnt, pred = accuracy(pred.cpu().numpy(),
                                             hm_gt.cpu().numpy()) 

        if avg_acc > val_acc.val:

            print(f"\rHighest Validation Accuracy! {avg_acc:.5f}")
            torch.save(model.state_dict(), "save_model/torch_model.pt") 
        
        val_acc.update(avg_acc, cnt)

        print("Val_loss {loss.val:.5f} | Valid_accuracy_value {acc.val:.5f} | Valid_accuracy_average {acc.avg:.5f}"
                  .format(epoch, iter, len(val_loader), loss=val_losses, acc=val_acc))

        wandb.log({"Valid loss": val_losses.avg, "Valid accuracy value" : acc.val})
        wandb.log({"Table": show_img_wandb(img, hm_gt, pred_logit)})

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='lsp')
    parser.add_argument("--model", type=str)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--resize", type=str, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs",  type=int ,default=50)
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
    

    
