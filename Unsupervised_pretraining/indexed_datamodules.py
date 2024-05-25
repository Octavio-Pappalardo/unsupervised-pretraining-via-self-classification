from transformations import return_augmentations 
from utils import indexed_dataset

import torchvision
import torch.utils.data as data

import random
import lightning.pytorch as pl




#------------------------------Data modules------------------
'''This datamodules have as a training dataset a collection of images with their corresponding index (which acts as a label) .Their validation and test sets
are subsets of the training dataset. This is necessary because the head of the self classifier model has to be retrained for each dataset'''



class STL10DataModule(pl.LightningDataModule):
    def __init__(self,batch_size,data_dir: str = "../data",num_workers=0,validation_size=20000):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size=batch_size
        self.validation_size=validation_size
        self.num_workers=num_workers

        self.train_set_size=100000
        self.image_height_width=96 
        self.transforms=return_augmentations(self.image_height_width)

        self.pesistent_workers=True if num_workers>0 else False

    def prepare_data(self):
        # download and other one time processes
        
        torchvision.datasets.STL10(root=self.data_dir,split="unlabeled",download=True,transform=None)
 
    def setup(self, stage: str=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" :

            #100 000 unlabeled 96x96 images used for unsupervised pretraining  .it contains a similar but broader distribution of images than the evaluation dataset
            STL10_unlabeled_dataset = torchvision.datasets.STL10(root = self.data_dir,
                split="unlabeled",
                download=False,
                transform=None,
            )

            self.train_dataset=indexed_dataset(STL10_unlabeled_dataset,self.transforms)

            #choose a subset of the training dataset to use also as validation. 
            dataset_indices=range(len(self.train_dataset))  #[s[1] for s in dataset]
            #randomly select validation_size elements from the list of indices
            validation_indices=random.sample(dataset_indices, self.validation_size)
            #create subset
            self.val_dataset = data.Subset(self.train_dataset, validation_indices)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" :
            dataset = torchvision.datasets.STL10(root = self.data_dir,split="unlabeled", download=False,transform=None)
            self.test_dataset = indexed_dataset(dataset, self.transforms)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                shuffle=True,drop_last=True,pin_memory=True,num_workers=self.num_workers,persistent_workers=self.pesistent_workers)
    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size,
                                shuffle=False,drop_last=False,pin_memory=True,num_workers=self.num_workers,persistent_workers=self.pesistent_workers)
    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                shuffle=False,drop_last=True,pin_memory=True,num_workers=self.num_workers,persistent_workers=self.pesistent_workers)
    