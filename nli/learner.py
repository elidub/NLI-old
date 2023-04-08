# define the LightningModule
import torch
from torch import optim, nn
import pytorch_lightning as pl

class NLINet(pl.LightningModule):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.loss_fn = nn.CrossEntropyLoss()

    def concat_sentreps(self, sentrep1, sentrep2):
        return torch.cat([sentrep1, sentrep2, torch.abs(sentrep1 - sentrep2), sentrep1 * sentrep2], dim=1)

    def step(self, batch, mode='train'):
        s1, s2, y, e1, e2, len1, len2 = batch
        u, v = self.encoder(e1, len1), self.encoder(e2, len2)

        print('u, v', u.shape, v.shape)



        features = self.concat_sentreps(u, v)

        print('features', features.shape)

        # Loss
        y_hat = self.classifier(features)
        loss = self.loss_fn(y_hat, y)

        # Accuracy
        pred = torch.argmax(y_hat, dim=1)
        acc = torch.sum(pred == y) / len(y)

        # Log
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)

        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch, 'train')

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch, 'val')

    def test_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch, 'test')


    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.1, weight_decay=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=1)
        return {'optimizer': optimizer, 'scheduler': scheduler, 'monitor': 'val_loss'}

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr)

        if lr < 1e-5:
            self.trainer.should_stop = True