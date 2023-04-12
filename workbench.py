import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl

# Define the PyTorch Lightning module
class MNISTClassifier(pl.LightningModule):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, threshold = 0.1, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr)

        print(f"Epoch {self.current_epoch} lr: {lr} val_loss: {self.trainer.callback_metrics['val_loss']}")



# Define the PyTorch Lightning dataloader
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super(MNISTDataModule, self).__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        MNIST(root='data/', train=True, download=True)
        MNIST(root='data/', train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MNIST(root='data/', train=True, transform=ToTensor())
            self.val_dataset = MNIST(root='data/', train=False, transform=ToTensor())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


# Initialize the model, data module, and trainer
model = MNISTClassifier()
data_module = MNISTDataModule()
trainer = pl.Trainer(gpus=1, max_epochs=10)

# Train the model
trainer.fit(model, datamodule=data_module)
