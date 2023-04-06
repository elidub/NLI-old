import urllib.request
import zipfile
import os
from datasets import load_dataset

import nltk
nltk.download('punkt')


def preprocess(example):
    example['premise']    = [word.lower() for word in nltk.tokenize.word_tokenize(example['premise'])]
    example['hypothesis'] = [word.lower() for word in nltk.tokenize.word_tokenize(example['hypothesis'])]
    return example

def download_snli():
    print('Downloading SNLI dataset...')
    dataset_snli = load_dataset("snli")

    print('Pre-processing SNLI dataset...')
    for split in dataset_snli:
        dataset_snli[split] = dataset_snli[split].map(preprocess)

    print('Saving SNLI dataset to disk...')
    dataset_snli.save_to_disk("data/snli")

    print('Done!')


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

if __name__ == '__main__':

    # make sure the data directory exists
    os.makedirs('data', exist_ok=True)

    download_snli()
    # download_glove()