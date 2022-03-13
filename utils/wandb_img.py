import wandb
import numpy as np
import cv2
import copy

def show_img_wandb(img, hm):

    hm = hm.detach().cpu().numpy()

    my_data = []
    
    for i in range(4):

        data = [wandb.Image(img[i])]

        for j in range(14):

            data.append(wandb.Image(hm[i,j,:,:]))
        my_data.append(data)

    table = wandb.Table(data=my_data, columns = ["Image", 
                                "0", "1", "2", "3", "4", "5",
                                "6", "7", "8", "9", "10", "11", 
                                "12","13"])
    
    return table

