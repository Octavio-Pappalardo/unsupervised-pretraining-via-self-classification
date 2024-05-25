import sys
sys.path.append('../Unsupervised_pretraining')

import torch.nn as nn
import torchvision
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os
from copy import deepcopy


from evaluation_datamodules import labeled_STL10DataModule
from evaluation_modules import Linear_head_classifier

from Sc_module import Self_classifier


def train(use_trained_encoder=True ,train_encoder=False):
    #-----------------------------------------------------------------------

    NUM_WORKERS = 4 # os.cpu_count()
    batch_size= 256
    data_dir= '../../data'

    #instantiate data_module
    data_module=labeled_STL10DataModule(batch_size=batch_size, data_dir=data_dir,num_workers=NUM_WORKERS )



    #-------------------------------------------------------------

    if use_trained_encoder==True:
        #LOAD trained ENCODER
        weights_filename='../Unsupervised_pretraining/weights_and_hparams/Self_classifier/file_name.ckpt'

        Sc_model = Self_classifier.load_from_checkpoint(f"{weights_filename}",hidden_dim=200,num_training_examples=100000) #arguments must be set to match the architecture of the trained weights

        encoder = deepcopy(Sc_model.full_self_classifier)
        # keep only the encoder part of the pretrained self classifying network 
        encoder = list(encoder.children())[0]

    
    elif use_trained_encoder==False:
        #instantiate untrained encoder (for comparing performance)
        encoder = torchvision.models.resnet18(weights=None, num_classes=8)  # num_classes is ignored
        encoder.fc=nn.Identity() #remove last layer

    if train_encoder==False:
        #controls wether the encoder backbone is also finetuned or just the linear head
        encoder.requires_grad_(False)

    #instantiate model
    embedding_dim=512 #must match encoder final dim
    lr=1e-4
    model=Linear_head_classifier(encoder=encoder,embedding_dim=embedding_dim, num_classes=data_module.num_classes, lr=lr)



    #----------------------------------------------------------------------
    ### Configure and Instantiate trainer ###

    #Set location for logs
    TENSORBOARD_LOGS_PATH='./tensorboard_logs'
    WEIGHTS_HPARAMS_PATH='./weights_and_hparams'

    #instantiate the logger to be used in training
    tensorboard_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(TENSORBOARD_LOGS_PATH, "Evaluator"))

    # define the behaviour of model checkpointing
    checkpoint_callback1 = ModelCheckpoint(
        save_weights_only=False,
        dirpath= os.path.join(WEIGHTS_HPARAMS_PATH, "Evaluator"),
        filename="sc-{epoch:02d}-{val_acc:.2f}"+f'_train_encoder={train_encoder}',
        mode="max", 
        save_top_k=1,
        monitor="val_acc")
    checkpoint_callback2 = ModelCheckpoint(
        save_weights_only=False,
        dirpath= os.path.join(WEIGHTS_HPARAMS_PATH, "Evaluator"),
        filename="sc-{epoch:02d}-{val_loss:.2f}"+f'_train_encoder={train_encoder}',
        mode="min", 
        save_top_k=1,
        monitor="val_loss")

    #define wether to stop training earlier
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=8, verbose=False, mode="max")


    Self_classifier_trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=500,
        logger=tensorboard_logger, 
        check_val_every_n_epoch=10,
        log_every_n_steps=20, 
        callbacks=[
            checkpoint_callback1,
            checkpoint_callback2,
            early_stop_callback
        ],
    )

    #------------------------
    ### Train ###
    Self_classifier_trainer.fit(model=model,datamodule=data_module)


if __name__ == '__main__':
    train(use_trained_encoder=True ,train_encoder=False)