# define the LightningModule
import torch
from torch import optim, nn
import pytorch_lightning as pl

class Learner(pl.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.save_hyperparameters(ignore=['net'])

        self.net = net
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_acc = None

    def encode(self, s):
        e = self.net.encode(s)
        return e

    def step(self, batch, mode='train'):
        s1, s2, y, e1, e2, len1, len2 = batch

        # Forward
        y_hat = self.net(s1, s2, len1, len2)

        # Loss
        loss = self.loss_fn(y_hat, y)

        # Accuracy
        pred = torch.argmax(y_hat, dim=1)
        acc = torch.sum(pred == y) / len(y)

        # Log
        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_acc",  acc,  prog_bar=True)

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
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor=0.2, patience=1, verbose=True)
        # return {
        #     'optimizer': optimizer, 
        #     'lr_scheduler': scheduler,
        #     'monitor': 'val_acc',
        #     # 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch', 'frequency': 1},
        # }
        return optimizer


    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        val_acc = self.trainer.callback_metrics['val_acc']

        new_lr = lr * 0.99

        # if val_acc is lower than previous epoch, reduce lr
        if self.val_acc is not None and val_acc < self.val_acc:
            print('Reducing learning rate')
            new_lr = new_lr / 5
        self.val_acc = val_acc

        self.log('lr', new_lr, prog_bar=True)

        self.trainer.optimizers[0].param_groups[0]['lr'] = new_lr
        
        print(f'Epoch {self.current_epoch} old lr: {lr} new lr: {new_lr} val_acc: {val_acc}')

        if lr < 1e-5:
            self.trainer.should_stop = True