import urllib.request
import zipfile
import os
from datasets import load_dataset, load_from_disk
import numpy as np
from tqdm import tqdm
import pickle
import torch
import nltk
nltk.download('punkt')




####### Download and pre-process the SNLI dataset #######

def tokenize(example):
    example['premise']    = [word.lower() for word in nltk.tokenize.word_tokenize(example['premise'])]
    example['hypothesis'] = [word.lower() for word in nltk.tokenize.word_tokenize(example['hypothesis'])]
    return example

def drop_missing_label(example):
    return example['label'] != -1

def download_snli():
    print('Downloading SNLI dataset...')
    dataset_snli = load_dataset("snli")

    print('Pre-processing SNLI dataset...')
    for split in dataset_snli:
        dataset_snli[split] = dataset_snli[split].map(tokenize)
        dataset_snli[split] = dataset_snli[split].filter(drop_missing_label)

    print('Saving SNLI dataset to disk...')
    dataset_snli.save_to_disk("data/snli")

    print('Done!')





####### Download the GloVe dataset #######

def download_glove():
    print("Downloading GloVe dataset...")
    glove_url = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
    zip_file_name = "data/glove.840B.300d.zip"

    urllib.request.urlretrieve(glove_url, zip_file_name)

    # Extract the GloVe dataset in the data directory and remove zip file
    print("Extracting GloVe dataset...")
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall("data")
    os.remove(zip_file_name)

    print("Done!")


####### Create word vectors and vocab #######


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

def create_wordvec():
    dataset_snli = load_from_disk('data/snli')

    print('Building vocab and word vectors...')
    vocab = get_vocab(dataset_snli['train'])
    wordvec = get_glove(vocab, glove_path = "data/glove.840B.300d.txt")

    print(f'Vocab size: {len(vocab)}')
    print(f'Word vector size: {len(wordvec)}')

    with open('store/wordvec.pkl', 'wb') as f:
        pickle.dump(wordvec, f)






if __name__ == '__main__':

    # make sure the data directory exists
    os.makedirs('data', exist_ok=True)

    # download_snli()
    # download_glove()
    create_wordvec()