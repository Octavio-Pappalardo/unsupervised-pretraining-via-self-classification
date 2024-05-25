
import torch.nn as nn
from torchvision import transforms as T
import torch
import random



class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


def return_augmentations(image_height_width):
    #should be modified if images are not square
    augmentations= torch.nn.Sequential(
                RandomApply(
                    T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                    p = 0.4
                ),
                T.RandomGrayscale(p=0.12),
                T.RandomHorizontalFlip(),
                RandomApply(T.RandomRotation(35) ,p=0.22),
                RandomApply(
                    T.GaussianBlur((int(0.1 * image_height_width)), (0.1, 2.0)),
                    p = 0.2
                ),
                T.RandomResizedCrop((image_height_width), (0.7,1.0),antialias=True),
                T.Normalize(
                    mean=torch.tensor([0.485, 0.456, 0.406]),
                    std=torch.tensor([0.229, 0.224, 0.225])),
                    
            )
    return T.Compose([T.ToTensor(),augmentations])

