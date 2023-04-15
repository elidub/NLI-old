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

from data import NLIDataModule, DataSetPadding
from models import AvgWordEmb, UniLSTM, BiLSTM, MaxPoolLSTM, MLP, NLINet
from learner import Learner
from train import setup_vocab, setup_model
import argparse
import os

# import SentEval
PATH_TO_SENTEVAL = './'
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def parse_option():
    parser = argparse.ArgumentParser(description="Training NLI models")

    parser.add_argument('--model_type', type=str, default='uni_lstm', help='Model type', choices=['avg_word_emb', 'uni_lstm', 'bi_lstm', 'max_pool_lstm'])
    parser.add_argument('--ckpt_path', type=str, default = 'lightning_logs/version_2/checkpoints/epoch=2-step=48.ckpt', help='Path to save checkpoint')
    parser.add_argument('--version', default= 'version_0', help='Version of the model to load')

    parser.add_argument('--path_to_results', type=str, default='store/results.txt', help='Path to results')
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

    ckpt_path = params.ckpt_path
    versions = os.listdir(os.path.join('logs', ckpt_path))
    version = versions[0] if len(versions) == 1 else params.version
    ckpts = os.listdir(os.path.join('logs', ckpt_path, version, 'checkpoints'))
    assert len(ckpts) == 1
    ckpt_path = os.path.join('logs', ckpt_path, version, 'checkpoints', ckpts[0])

    logging.info(f'Loading model from {ckpt_path}')

    vocab  = setup_vocab(params['path_to_vocab'])
    _, net = setup_model(model_type = params['model_type'], vocab = vocab)
    params.model = Learner.load_from_checkpoint(ckpt_path, net=net)
    params.model.eval()

    params.prep_sent = lambda sent: DataSetPadding(None, vocab).prepare_sent(sent)

    logging.info(f'Evaluating!')

    return

def batcher(params, batch):
    """
    In this example we use the average of word embeddings as a sentence representation.
    Each batch consists of one vector for sentence.
    Here you can process each sentence of the batch, 
    or a complete batch (you may need masking for that).
    
    """
    batch = [sent if sent != [] else ['.'] for sent in batch]

    # sent_ids, slens = zip(*[params.prep_sent(sent) for sent in batch])
    # sent_ids = torch.stack(sent_ids)
    # slens    = torch.tensor(slens)

    # embeddings = params.model.net.encode(sent_ids, slens)
    # embeddings = embeddings.detach().cpu().numpy()

    embeddings = np.random.randn(len(batch), 300)

    return embeddings


def main(args):
    # we use logistic regression (usepytorch: Fasle) and kfold 10
    # In this dictionary you can add extra information that you model needs for initialization
    # for example the path to a dictionary of indices, of hyper parameters
    # this dictionary is passed to the batched and the prepare fucntions
    params_senteval = {
        'task_path': args.path_to_data, 'usepytorch': False, 'kfold': 2,
        'model_type': args.model_type,
        'ckpt_path' : args.ckpt_path,
        'version'   : args.version,
        'path_to_vocab': args.path_to_vocab,
    }

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                      'MRPC', 'SICKEntailment', 'STS14']
    
    transfer_tasks = ['MR', 'CR']
    
    results = se.eval(transfer_tasks)
    print(results)

    # save results to txt file
    with open(args.path_to_results, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    args = parse_option()
    print(args)
    main(args)