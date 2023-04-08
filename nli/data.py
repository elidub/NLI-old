from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
import pickle
import torch

def get_vocab(dataset):
    vocab = set()
    for example in dataset:
        vocab.update(example['premise'])
        vocab.update(example['hypothesis'])
    return vocab

def get_glove(vocab, glove_path):

    with open(glove_path, "r", encoding="utf8") as f:
        lines = f.readlines()

    wordvec = {}
    for line in tqdm(lines):
        word, vec = line.split(' ', 1)
        if word in vocab:
            wordvec[word] = torch.tensor(list(map(float, vec.split())))

    wordvec['<unk>'] = torch.normal(mean=0, std=1, size=(300,))
    # wordvec['<pad>'] = torch.normal(mean=0, std=1, size=(300,))
    wordvec['<pad>'] = torch.zeros(300)

    return wordvec


if __name__ == '__main__':

    dataset_snli = load_from_disk('data/snli')

    print('Building vocab and word vectors...')
    vocab = get_vocab(dataset_snli['train'])
    wordvec = get_glove(vocab, glove_path = "data/glove.840B.300d.txt")

    print(f'Vocab size: {len(vocab)}')
    print(f'Word vector size: {len(wordvec)}')

    with open('store/wordvec.pkl', 'wb') as f:
        pickle.dump(wordvec, f)