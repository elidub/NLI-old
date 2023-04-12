# define the LightningModule
import torch
from torch import optim, nn
import pytorch_lightning as pl

class Learner(pl.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.CrossEntropyLoss()


    def step(self, batch, mode='train'):
        s1, s2, y, e1, e2, len1, len2 = batch

        # Forward
        y_hat = self.net(e1, e2, len1, len2)

        # Loss
        loss = self.loss_fn(y_hat, y)

        # Accuracy
        pred = torch.argmax(y_hat, dim=1)
        acc = torch.sum(pred == y) / len(y)

        # Log
        self.log(f"{mode}_loss", loss, prog_bar=True)
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
        optimizer = optim.Adam(self.parameters(), lr=0.1, weight_decay=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=1, verbose=True)
        return {
            'optimizer': optimizer, 
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
            # 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch', 'frequency': 1},
        }

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr)

        print(f"Epoch {self.current_epoch} lr: {lr} val_loss: {self.trainer.callback_metrics['val_loss']}")

        if lr < 1e-5:
            self.trainer.should_stop = True