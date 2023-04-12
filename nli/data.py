import torch
from datasets import load_from_disk
import pytorch_lightning as pl
import pickle
from torch import utils

class DataSetPadding():
    def __init__(self, dataset, wordvec, max_length=100):
        self.dataset = dataset
        self.wordvec = wordvec
        self.max_length = max_length

    def __len__(self):
        assert self.dataset.num_rows == len(self.dataset)
        return len(self.dataset)

    def get_embedding(self, sent):
        return torch.stack([self.wordvec[word] for word in sent])
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        s1, s2, y = example['premise'], example['hypothesis'], example['label']

        s1 = [word if word in self.wordvec else '<unk>' for word in s1]
        s2 = [word if word in self.wordvec else '<unk>' for word in s2]

        len1, len2 = len(s1), len(s2)

        if (len1 > self.max_length) or (len2 > self.max_length):
            raise ValueError('Sentence length exceeds max length')

        s1.extend(['<pad>'] * (self.max_length - len1))
        s2.extend(['<pad>'] * (self.max_length - len2))

        e1 = self.get_embedding(s1)
        e2 = self.get_embedding(s2)

        return s1, s2, y, e1, e2, len1, len2
    

class NLIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=1):
        super(NLIDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        with open('store/wordvec.pkl', 'rb') as f:
            wordvec = pickle.load(f)
        dataset_snli = load_from_disk('data/snli')

        self.dataset = {}
        for split, shuffle in zip(['train', 'validation'], [True, False]):
            self.dataset[split] = DataSetPadding(dataset_snli[split], wordvec)

            if split in ['train', 'validation']: self.dataset[split] = utils.data.Subset(self.dataset[split], range(1000))

    def train_dataloader(self):
        return utils.data.DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return utils.data.DataLoader(self.dataset['validation'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return utils.data.DataLoader(self.dataset['test'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    

if __name__ == '__main__':
    dataset_snli = load_from_disk('data/snli')

    for split in dataset_snli:
        l1, l2 = 0, 0
        for example in dataset_snli[split]:
            l1 = max(l1, len(example['premise']))
            l2 = max(l2, len(example['hypothesis']))
        print(f'{split}: {l1}, {l2}')