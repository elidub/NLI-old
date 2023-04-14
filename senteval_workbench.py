from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging


# Set PATHs
PATH_TO_SENTEVAL = './'
PATH_TO_DATA = './SentEval/data'
PATH_TO_VEC = './data/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

from nli.preprocess import create_dictionary, get_glove as get_wordvec

# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(params.word2id, PATH_TO_VEC)
    params.wvec_dim = 300
    return

def batcher(params, batch):
    print('batcher')
    print('params',params)
    print('batch', batch)
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings

def batcher(params, batch):

    batch = [sent if sent != [] else ['.'] for sent in batch]

    print(batch)

    # for sent in batch:
    #     for word in sent:
    #         print(word)
    #     print('---')

    assert False

    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                      'MRPC', 'SICKEntailment', 'STS14']
    results = se.eval(transfer_tasks)
    print(results)