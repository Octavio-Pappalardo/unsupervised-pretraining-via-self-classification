import sys
sys.path.append('../Unsupervised_pretraining')

import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import lightning.pytorch as pl


class Linear_head_classifier(pl.LightningModule):
    def __init__(self, encoder,embedding_dim, num_classes, lr, weight_decay=1e-3, max_epochs=100):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])

        self.encoder=encoder

        # Linear mapping from encoding to classes
        self.linear_head = nn.Linear(embedding_dim, num_classes)


    def configure_optimizers(self):
        optimizer= optim.SGD(self.linear_head.parameters(), lr=self.hparams.lr,momentum=0.95)
        return [optimizer]


    def _calculate_loss(self, batch, mode="train"):
        images, labels = batch
        encodings= self.encoder(images)

        logits = self.linear_head(encodings)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        self.log(mode + "_loss", loss)
        self.log(mode + "_acc", acc,prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")