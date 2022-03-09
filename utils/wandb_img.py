import wandb
import numpy as np

def show_img_wandb(img, gt, preds):

    img = img.detach().cpu().numpy()
    img = np.moveaxis(img, 1, -1)

    ground_truth = gt.detach().cpu().numpy()
    ground_truth = np.sum(ground_truth, axis=1)

    preds = preds[0].detach().cpu().numpy()
    preds = np.sum(preds, axis=1)

    my_data = []
    
    for i in range(4):

        data = [wandb.Image(img[i]), 
                #wandb.Image(ground_truth[i]),
                wandb.Image(preds[i])]  

        my_data.append(data)

    table = wandb.Table(data=my_data, columns = ["Image", "Prediction"])
    
    return table

    
