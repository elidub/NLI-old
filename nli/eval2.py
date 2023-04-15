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
import torch

from data import NLIDataModule, Vocabulary, DataSetPadding
from models import AvgWordEmb, UniLSTM, BiLSTM, MaxPoolLSTM, MLP, NLINet
from learner import Learner

# Set PATHs
PATH_TO_SENTEVAL = './'
PATH_TO_DATA = './SentEval/data'
PATH_TO_VEC = './data/glove.840B.300d.txt'


# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval



def prepare(params, samples):
    """
    In this example we are going to load Glove, 
    here you will initialize your model.
    remember to add what you model needs into the params dictionary
    """

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
    """
    In this example we use the average of word embeddings as a sentence representation.
    Each batch consists of one vector for sentence.
    Here you can process each sentence of the batch, 
    or a complete batch (you may need masking for that).
    
    """
    batch = [sent if sent != [] else ['.'] for sent in batch]

    sent_ids, slens = zip(*[params.prep_sent(sent) for sent in batch])
    sent_ids = torch.stack(sent_ids)
    slens    = torch.tensor(slens)

    embeddings = params.model.net.encode(sent_ids, slens)
    embeddings = embeddings.detach().cpu().numpy()

    return embeddings


# we use logistic regression (usepytorch: Fasle) and kfold 10
# In this dictionary you can add extra information that you model needs for initialization
# for example the path to a dictionary of indices, of hyper parameters
# this dictionary is passed to the batched and the prepare fucntions
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                      'MRPC', 'SICKEntailment', 'STS14']
    
    results = se.eval(transfer_tasks)
    print(results)