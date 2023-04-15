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
from data import Vocabulary
import argparse
import logging


def parse_option():
    parser = argparse.ArgumentParser(description="Training NLI models")

    parser.add_argument('--download_snli', action='store_true', default = False, help='Download and pre-process the SNLI dataset')
    parser.add_argument('--download_glove', action='store_true', default = False, help='Download the GloVe dataset')
    parser.add_argument('--create_vocab', action='store_true', default = False, help='Create the vocabulary')
    
    parser.add_argument('--path_to_vocab', type=str, default='store/vocab.pkl', help='Path to vocab')
    parser.add_argument('--path_to_vec', type=str, default='data/glove.840B.300d.txt', help='Path to GloVe dataset')

    args = parser.parse_args()

    return args


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

####### Main #######

def main(args):
    os.makedirs('data', exist_ok=True)
    
    if args.download_snli:
        download_snli()

    if args.download_glove:
        download_glove()
    
    if args.create_vocab:
        logging.info('Loading SNLI...')
        dataset_snli = load_from_disk('data/snli')

        samples = dataset_snli['train']['premise'] + dataset_snli['train']['hypothesis']
        
        vocab = Vocabulary(samples, path_to_vec = args.path_to_vec)

        with open(args.path_to_vocab, 'wb') as f:
            pickle.dump(vocab, f)

if __name__ == '__main__':
    args = parse_option()
    print(args)
    main(args)