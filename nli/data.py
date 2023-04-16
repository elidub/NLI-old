import torch
from datasets import load_from_disk
import pytorch_lightning as pl
import pickle
from torch import utils
from tqdm import tqdm

class Vocabulary:

    def __init__(self, samples, path_to_vec):

        dataset_corpus = self.get_words(samples)
        self.wordvec = self.get_wordvec(dataset_corpus, path_to_vec)
        self.id2word, self.word2id = self.create_dictionary(self.wordvec)

        print(f'Words in sample: {len(dataset_corpus)}\nWords in wordvec: {len(self.wordvec)}\nOverlapping words: {len(self.id2word)}')

    def get_words(self, sentences, threshold=0):
        words = {}
        for s in tqdm(sentences, desc = 'Creating dictionary'):
            for word in s:
                words[word] = words.get(word, 0) + 1

        if threshold > 0:
            newwords = {}
            for word in words:
                if words[word] >= threshold:
                    newwords[word] = words[word]
            words = newwords
        
        words['<s>'] = 1e9 + 4
        words['</s>'] = 1e9 + 3
        words['<p>'] = 1e9 + 2

        sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
        sorted_words = [w for (w, _) in sorted_words]
        
        return sorted_words

    def get_wordvec(self, dataset_corpus, path_to_vec):

        wordvec = {}
        wordvec['<unk>'] = torch.normal(mean=0, std=1, size=(300,))
        # wordvec['<pad>'] = torch.normal(mean=0, std=1, size=(300,))
        wordvec['<pad>'] = torch.zeros(300)

        # i = 0
        with open(path_to_vec, "r", encoding="utf8") as f:
            for line in tqdm(f, desc = f'Creating word vectors from {path_to_vec}', total = 2196017):
                word, vec = line.split(' ', 1)
                if word in dataset_corpus:
                    wordvec[word] = torch.tensor(list(map(float, vec.split())))
                # i += 1
                # if i > 10000:
                #     pass

        assert list( wordvec.keys() )[:2] == ['<unk>', '<pad>']

        return wordvec
    
    def create_dictionary(self, words):
        id2word = []
        word2id = {}
        for i, w in enumerate(words):
            id2word.append(w)
            word2id[w] = i

        return id2word, word2id

class DataSetPadding():
    def __init__(self, dataset, vocab, max_length=100):
        self.dataset = dataset
        # self.wordvec = wordvec
        self.maxlen = max_length
        self.vocab = vocab

        self.unk_id = list(self.vocab.wordvec.keys()).index('<unk>')
        self.pad_id = list(self.vocab.wordvec.keys()).index('<pad>')

    def __len__(self):
        # assert self.dataset.num_rows == len(self.dataset)
        return len(self.dataset)

    # def get_embedding(self, sent):
    #     return torch.stack([self.wordvec[word] for word in sent])
    
    def prepare_sent(self, sent):

        slen = len(sent)

        if slen > self.maxlen:
            print(f'Sentence length exceeding {slen}')
            sent = sent[:self.maxlen]
            slen = self.maxlen

        assert slen <= self.maxlen, f"Sentence length exceeds the maximum length of {self.maxlen}"
        assert slen > 0, "Sentence length is 0"
        assert self.unk_id == 0, "Unknown token id is not 0"
        assert self.pad_id == 1, "Padding token id is not 1"
        
        indices = [self.vocab.word2id.get(word, self.unk_id) for word in sent] + [self.pad_id] * (self.maxlen - slen)
        # indices = [self.vocab.word2id[word] if (word in self.vocab.word2id and word in self.vocab.wordvec) else self.unk_id for word in sent] + [self.pad_id] * (self.maxlen - slen)
        indices = torch.tensor(indices, dtype=torch.long)

        # sent = [word if word in self.wordvec else '<unk>' for word in sent]
        # sent.extend(['<pad>'] * (self.max_length - length))
        # embedding = [self.wordvec for word in sent]

        return indices, slen
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        s1, s2, y = example['premise'], example['hypothesis'], example['label']

        (s1, len1), (s2, len2) = self.prepare_sent(s1), self.prepare_sent(s2)

        x = s1, s2, len1, len2

        return x, y
    

class NLIDataModule(pl.LightningDataModule):
    def __init__(self, vocab, batch_size=64, num_workers=1):
        super(NLIDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab = vocab

    def setup(self, stage=None):

        dataset_snli = load_from_disk('data/snli')

        self.dataset = { split : DataSetPadding(dataset_snli[split], self.vocab) for split in dataset_snli }

        # for split in ['train', 'validation']:
        #      self.dataset[split] = utils.data.Subset(self.dataset[split], range(1000))

    def train_dataloader(self):
        return utils.data.DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return utils.data.DataLoader(self.dataset['validation'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return utils.data.DataLoader(self.dataset['test'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return utils.data.DataLoader(self.dataset['predict'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

if __name__ == '__main__':
    dataset_snli = load_from_disk('data/snli')

    for split in dataset_snli:
        l1, l2 = 0, 0
        for example in dataset_snli[split]:
            l1 = max(l1, len(example['premise']))
            l2 = max(l2, len(example['hypothesis']))
        print(f'{split}: {l1}, {l2}')