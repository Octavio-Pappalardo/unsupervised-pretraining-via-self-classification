import sys
sys.path.append('../Unsupervised_pretraining')

from transformations import return_augmentations 

import torch
import torchvision
import torch.utils.data as data
from torchvision import transforms as T
import random
import lightning.pytorch as pl




#------------------------------Data modules------------------


class labeled_STL10DataModule(pl.LightningDataModule):
    def __init__(self,batch_size,data_dir: str = "../data",num_workers=0,training_datapoints_per_class=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.training_datapoints_per_class=training_datapoints_per_class

        self.image_height_width=96 
        self.labels=[0,1,2,3,4,5,6,7,8,9]
        self.num_classes=10

        self.train_transforms=return_augmentations(self.image_height_width)
        self.test_transforms = T.Compose([T.ToTensor(), T.Normalize(
                    mean=torch.tensor([0.485, 0.456, 0.406]),
                    std=torch.tensor([0.229, 0.224, 0.225]))   ])

        self.pesistent_workers=True if num_workers>0 else False

    def prepare_data(self):
        # download and other one time processes
        
        torchvision.datasets.STL10(root=self.data_dir,split="unlabeled",download=True,transform=None)
        torchvision.datasets.STL10(root=self.data_dir,split="test",download=True,transform=None)

    def setup(self, stage: str=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" :

            #5000 labeled training images(500 per class) for supervised training on a classification task
            STL10_labeled_train_dataset = torchvision.datasets.STL10(
                root = self.data_dir,
                split="train",
                download=False,
                transform=self.train_transforms,
            )
            VAL_PERCENTAGE=0.1 #we set the validation set to be 10% of the training set. this means 500 examples; 50 of each class

            #split the labeled part of the STL10 dataset meant for training into a training and a validation dataset (mantaining a balanced amount of examples per class in each dataset)
            self.train_dataset ,self.val_dataset= split_mantaining_class_balance(STL10_labeled_train_dataset,VAL_PERCENTAGE)

            #if num datapoints per class for training is specified ,the training dataset is reduced such that it satisfies this.
            if self.training_datapoints_per_class != None:
                num_examples_per_class_in_train_dataset=len(self.train_dataset)/self.num_classes
                assert self.training_datapoints_per_class<=num_examples_per_class_in_train_dataset, "not enough datapoints to create the specified datasets"

                self.train_dataset=subsetDataset_of_n_examples_per_class(self.train_dataset,self.training_datapoints_per_class,self.labels)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" :
            #8000 labeled test images(800 per class) for testing classification performance
            self.test_dataset = torchvision.datasets.STL10(
                root = self.data_dir,
                split="test",
                download=False,
                transform=self.test_transforms,
            )


    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                shuffle=True,drop_last=True,pin_memory=True,num_workers=self.num_workers,persistent_workers=self.pesistent_workers)
    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size,
                                shuffle=False,drop_last=False,pin_memory=True,num_workers=self.num_workers,persistent_workers=self.pesistent_workers)
    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                shuffle=False,drop_last=True,pin_memory=True,num_workers=self.num_workers,persistent_workers=self.pesistent_workers)
  



#---------- Helper functions -------------------------


def subsetDataset_of_n_examples_per_class(dataset,num_examples,labels):
    '''returns a subset of dataset with num_examples data
    points for each label in labels(assuming such number of examples exist in the dataset).'''
    #dataset indices that will be kept
    indices=[]

    targets=[s[1] for s in dataset]
    for label in labels:
        # We create a tensor that has `True` at an index if the sample belongs to label
        mask = torch.tensor(targets) == label
        #1d tensor with the indices of the datapoints that satisfies the requirement
        mask_indices= mask.nonzero().reshape(-1)
        #randomly select n elements from the list of indices
        mask_indices=random.sample(mask_indices.tolist(), num_examples)
        #add the indices to the total list of indices to keep
        indices.extend(mask_indices)

    indices=torch.tensor(indices)
    # Then we use this indices to obtain the desired subset
    dataset_subset = data.Subset(dataset, indices)
    return dataset_subset

def split_mantaining_class_balance(dataset,val_percentage=0.1):
    '''separates a labeled dataset into a training dataset and validation dataset
    , putting in the validation dataset an equal amount of examples of each class.
    '''
    assert val_percentage<1 

    targets=[s[1] for s in dataset]#list with each datapoints label
    labels=set(targets) #set with each possible label
    num_examples_per_class= int(  len(dataset)*val_percentage/len(labels)  )

    #-
    #dataset indices that will go to the validation dataset
    val_indices=[]
    for label in labels:
        # We create a tensor that has `True` at an index if the sample belongs to label
        mask = torch.tensor(targets) == label
        #1d tensor with the indices of the datapoints that satisfies the requirement
        mask_indices= mask.nonzero().reshape(-1)
        #randomly select n elements from the list of indices
        mask_indices=random.sample(mask_indices.tolist(), num_examples_per_class)
        #add the indices to the total list of indices to keep
        val_indices.extend(mask_indices)

    val_indices_tensor= torch.tensor(val_indices)
    # We then pass the original dataset and the indices we are interested in
    val_dataset =data.Subset(dataset, val_indices_tensor)

    #----We now obtain the training dataset
    val_indices=set(val_indices)
    train_indices=[i for i in range(len(dataset)) if i not in val_indices]
    train_indices=torch.tensor(train_indices)
    train_dataset=data.Subset(dataset,train_indices)
    #--
    return train_dataset ,val_dataset

