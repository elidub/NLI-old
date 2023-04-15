from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging

import pickle

import torch
import pytorch_lightning as pl
import argparse

from data import NLIDataModule, Vocabulary, DataSetPadding
from models import AvgWordEmb, UniLSTM, BiLSTM, MaxPoolLSTM, MLP, NLINet
from learner import Learner


import pickle
from datasets import load_from_disk

# Set PATHs
PATH_TO_SENTEVAL = './'
PATH_TO_DATA = './SentEval/data'
PATH_TO_VEC = './data/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):

    print('samples len', len(samples))

    ckpt_path = 'lightning_logs/version_2/checkpoints/epoch=2-step=48.ckpt'


    vocab = pickle.load(open('store/vocab.pkl', 'rb'))

    hidden_dim = 2048

    encoder = UniLSTM(hidden_dim)
    classifier = MLP(hidden_dim*4)

    net = NLINet(encoder, classifier, vocab)
    model = Learner(net)

    params.model = Learner.load_from_checkpoint(ckpt_path, net=net)
    params.model.eval();

    params.prep_sent = lambda sent: DataSetPadding(None, vocab).prepare_sent(sent)
    return



def batcher(params, batch):

    # print('batch', len(batch))s


    # sent_ids, slens = zip(*[params.prep_sent(sent) for sent in batch])
    # sent_ids = torch.stack(sent_ids)
    # slens    = torch.tensor(slens)

    # embeddings = params.model.net.encode(sent_ids, slens)

    # embeddings = embeddings.detach().cpu().numpy()

    # print(embeddings.shape)

    embeddings = np.random.randn(len(batch), 2048)

    return embeddings




# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10}
# params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
#                                  'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                      'MRPC', 'SICKEntailment', 'STS14']
    results = se.eval(transfer_tasks)
    print(results)