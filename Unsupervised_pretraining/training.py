
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import os


from indexed_datamodules import STL10DataModule
from Sc_module import Self_classifier


def train():
    #-----------------------------------------------------------------------
    data_dir= '../../data'
    
    # Define run hyperparameters
    NUM_WORKERS = 0 # os.cpu_count()
    batch_size= 512
    validation_size= 15000
    HIDDEN_DIMENSION=200 #dim of the intermediate layer between the encoding and the predictions of the self classifier
    lr=1e-5

    #instantiate data_module
    data_module=STL10DataModule(batch_size=batch_size,data_dir=data_dir,num_workers=NUM_WORKERS ,validation_size=validation_size)

    #instantiate model
    model=Self_classifier(hidden_dim=HIDDEN_DIMENSION , num_training_examples=data_module.train_set_size  ,lr=lr)

    #----------------------------------------------------------------------
    ### Configure and Instantiate trainer ###

    #Set location for logs
    TENSORBOARD_LOGS_PATH='./tensorboard_logs'
    WEIGHTS_HPARAMS_PATH='./weights_and_hparams'


    #instantiate the logger to be used in training
    tensorboard_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(TENSORBOARD_LOGS_PATH, "Self_classifier"))


    # define the behaviour of model checkpointing
    checkpoint_callback1 = ModelCheckpoint(
        dirpath= os.path.join(WEIGHTS_HPARAMS_PATH, "Self_classifier"),
        filename="self-classifier-{epoch:02d}-{validation_acc_top20:.2f}",
        mode="max", 
        save_top_k=1,
        monitor="validation_acc_top20")
    checkpoint_callback2 = ModelCheckpoint(
        dirpath= os.path.join(WEIGHTS_HPARAMS_PATH, "Self_classifier"),
        filename="self-classifier-{epoch:02d}-{validation_acc_top5:.2f}",
        mode="max", 
        save_top_k=1,
        monitor="validation_acc_top5")
    checkpoint_callback3 = ModelCheckpoint(
        dirpath= os.path.join(WEIGHTS_HPARAMS_PATH, "Self_classifier"),
        filename="self-classifier-{epoch:02d}-{validation_acc_top100:.2f}",
        mode="max", 
        save_top_k=1,
        monitor="validation_acc_top100")
    checkpoint_callback4 = ModelCheckpoint(
        dirpath= os.path.join(WEIGHTS_HPARAMS_PATH, "Self_classifier"),
        filename="self-classifier-{epoch:02d}-{validation_acc_top1000:.2f}",
        mode="max", 
        save_top_k=1,
        monitor="validation_acc_top1000")
    checkpoint_callback5 = ModelCheckpoint(
        dirpath= os.path.join(WEIGHTS_HPARAMS_PATH, "Self_classifier"),
        filename="self-classifier-{epoch:02d}-{val_loss:.2f}",
        mode="min", 
        save_top_k=1,
        monitor="val_loss")

    #define wether to stop training earlier if a certain metric doesnt improve for a certain amount of consecutive epochs
    early_stop_callback = EarlyStopping(monitor="validation_acc_top20", min_delta=0.00, patience=5, verbose=False, mode="max")


    Self_classifier_trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=2000,
        logger=tensorboard_logger, 
        check_val_every_n_epoch=5, 
        callbacks=[
            checkpoint_callback1,
            checkpoint_callback2,
            checkpoint_callback3,
            checkpoint_callback4,
            checkpoint_callback5,
            early_stop_callback
        ],
    )

    #------------------------
    ### Train ###

    Self_classifier_trainer.fit(model=model,datamodule=data_module)





if __name__ == '__main__':
    train()