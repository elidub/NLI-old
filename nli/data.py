import torch
from datasets import load_from_disk

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
    

if __name__ == '__main__':
    dataset_snli = load_from_disk('data/snli')

    for split in dataset_snli:
        l1, l2 = 0, 0
        for example in dataset_snli[split]:
            l1 = max(l1, len(example['premise']))
            l2 = max(l2, len(example['hypothesis']))
        print(f'{split}: {l1}, {l2}')