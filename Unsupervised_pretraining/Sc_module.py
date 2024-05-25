import utils

import torch
import torch.nn as nn
import torchvision
import lightning.pytorch as pl



class Self_classifier(pl.LightningModule):
    def __init__(self, hidden_dim, num_training_examples, lr):
        super().__init__()
        self.num_training_examples=num_training_examples
        self.save_hyperparameters(ignore=['hidden_dim','num_training_examples'])
        

        self.encoder = torchvision.models.resnet18(
            weights=None, num_classes=8
        )  #num_classes is is ignored
        self.encoder.fc=nn.Identity() #remove last layer
        #The image encoding usig resnet 18 (with the final linear layer removed) is of size 512
        
        # Self_classifier head: The MLP that classifies to which datapoint each encoding corresponds
        self.self_classifier_head = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear( hidden_dim, num_training_examples),
        )

        #combine the encoder and head into a single network
        self.full_self_classifier=nn.Sequential(self.encoder,self.self_classifier_head)

        #define the classification loss function
        self.criterion = nn.CrossEntropyLoss(reduction='mean',label_smoothing=0.1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


    def training_step(self, batch, batch_idx):
        imgs , indices = batch
        prediction_logits = self.full_self_classifier(imgs)
        loss = self.criterion(input=prediction_logits, target=indices) 

        if batch_idx % 50 == 0:
            # Logging ranking metrics
            topk_accuracies= utils.accuracy(output=prediction_logits, target=indices, topk=(1,5,20,50,100,1000))
            self.log('train'+'_loss',loss,prog_bar=True)
            self.log('train' + "_acc_top1", topk_accuracies[0])
            self.log('train' + "_acc_top5", topk_accuracies[1])
            self.log('train' + "_acc_top20", topk_accuracies[2])
            self.log('train' + "_acc_top50", topk_accuracies[3])
            self.log('train' + "_acc_top100", topk_accuracies[4])
            self.log('train' + "_acc_top1000", topk_accuracies[5])
        
        return loss



    def validation_step(self, batch, batch_idx):
        
        #log value of metrics over validation dataset (a mean is taken over all the batches by lightning)
        imgs , indices = batch
        prediction_logits= self.full_self_classifier(imgs)
        loss = self.criterion(input=prediction_logits, target=indices) 
        self.log("val_loss", loss,prog_bar=True)

        #log top k accuracies
        topk_accuracies= utils.accuracy(output=prediction_logits, target=indices, topk=(1,5,20,50,100,1000))
        self.log('validation' + "_acc_top1", topk_accuracies[0])
        self.log('validation' + "_acc_top5", topk_accuracies[1])
        self.log('validation' + "_acc_top20", topk_accuracies[2],prog_bar=True)
        self.log('validation' + "_acc_top50", topk_accuracies[3])
        self.log('validation' + "_acc_top100", topk_accuracies[4])
        self.log('validation' + "_acc_top1000", topk_accuracies[5])
