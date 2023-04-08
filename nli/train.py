from datasets import load_from_disk
import pickle
import torch
from torch import utils
import pytorch_lightning as pl

from data import DataSetPadding
from models import Baseline, MLP
from learner import NLINet

if __name__ == '__main__':

    with open('store/wordvec.pkl', 'rb') as f:
        wordvec = pickle.load(f)
    dataset_snli = load_from_disk('data/snli')


    dataloader = {}
    for split, shuffle in zip(['train', 'validation'], [True, False]):
        dataset = DataSetPadding(dataset_snli[split], wordvec)
        if split == 'train':
            dataset = utils.data.Subset(dataset, range(10000))
        dataloader[split] = utils.data.DataLoader(dataset, batch_size=64, shuffle=shuffle, num_workers=24)


    encoder = Baseline()
    classifier = MLP(300*4, 512, 3)
    model = NLINet(encoder, classifier)

    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(max_epochs=10, log_every_n_steps=1, accelerator=device)
    trainer.fit(model, train_dataloaders = dataloader['train'], val_dataloaders = dataloader['validation'])