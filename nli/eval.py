# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import sklearn
# import data
import pickle
import yaml
import json
import torch
import argparse
import os

# from nli.data import DataSetPadding
# from models import AvgWordEmb, UniLSTM, BiLSTM, MaxPoolLSTM, MLP, NLINet
# from learner import Learner
# from train import setup_vocab, setup_model
from setup import find_checkpoint, load_model, prep_sent
    

# import SentEval
PATH_TO_SENTEVAL = './'
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def parse_option():
    parser = argparse.ArgumentParser(description="Training NLI models")

    parser.add_argument('--model_type', type=str, default='uni_lstm', help='Model type', choices=['avg_word_emb', 'uni_lstm', 'bi_lstm', 'max_pool_lstm'])
    parser.add_argument('--ckpt_path', type=str, default = None, help='Path to save checkpoint')
    parser.add_argument('--version', default= 'version_0', help='Version of the model to load')

    # parser.add_argument('--path_to_results', type=str, default='store/results.txt', help='Path to results')
    parser.add_argument('--path_to_vocab', type=str, default='store/vocab.pkl', help='Path to vocab')
    parser.add_argument('--path_to_data', type=str, default = './SentEval/data', help='Path to data')


    args = parser.parse_args()

    return args


def prepare(params, samples):
    """
    In this example we are going to load Glove, 
    here you will initialize your model.
    remember to add what you model needs into the params dictionary
    """

    params.model, vocab = load_model(params['model_type'], params['path_to_vocab'], params['ckpt_path'], params['version'])
    params.prep_sent = lambda sent: prep_sent(sent, vocab)

    logging.info(f'Evaluating!')

    return

def batcher(params, batch):
    """
    In this example we use the average of word embeddings as a sentence representation.
    Each batch consists of one vector for sentence.
    Here you can process each sentence of the batch, 
    or a complete batch (you may need masking for that).
    
    """

    # Since the words are already tokenized, we only apply lowercasing
    # to have the same preprocessing steps as we did for the SNLI data
    batch = [[word.lower() for word in sent] if sent != [] else ['.'] for sent in batch]

    sent_ids, slens = zip(*[params.prep_sent(sent) for sent in batch])
    sent_ids = torch.stack(sent_ids)

    slens    = torch.tensor(slens)

    embeddings = params.model.net.encode(sent_ids, slens)
    embeddings = embeddings.detach().cpu().numpy()

    # embeddings = np.random.randn(len(batch), 300)

    return embeddings


def main(args):
    ckpt_path = args.ckpt_path if args.ckpt_path is not None else args.model_type

    # we use logistic regression (usepytorch: Fasle) and kfold 10
    # In this dictionary you can add extra information that you model needs for initialization
    # for example the path to a dictionary of indices, of hyper parameters
    # this dictionary is passed to the batched and the prepare fucntions
    params_senteval = {
        'task_path': args.path_to_data, 'usepytorch': True, 'kfold': 10,
        'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4},
        'model_type': args.model_type,
        'ckpt_path' : ckpt_path,
        'version'   : args.version,
        'path_to_vocab': args.path_to_vocab,
    }

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                      'MRPC', 'SICKEntailment', 'STS14']

    results = se.eval(transfer_tasks)
    print(results)

    # save results to txt file
    _, version_path = find_checkpoint(ckpt_path, args.version)
    with open(os.path.join(version_path, 'results.txt'), 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    args = parse_option()
    print(args)
    main(args)