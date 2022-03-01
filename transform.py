import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn

class Transform(nn.Module):

    def __init__(self, data_name='lsp', resize=128):

        data_means = {'lsp' : [0.3781, 0.4442, 0.4512]}
        data_std = {'lsp' : [0.2409, 0.2334, 0.2498]}

        mean = data_means[data_name]
        std = data_std[data_name]

        self.transforms = A.Compose([

                A.Resize(resize, resize),
                A.Normalize(mean, std),
                ToTensorV2()
                ]) 

    def __call__(self, image):

        return self.transforms(image=image)
