import pickle
from models import AvgWordEmb, UniLSTM, BiLSTM, MaxPoolLSTM, MLP, NLINet
from learner import Learner
import os
import logging

from data import DataSetPadding

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'Vocabulary':
            from data import Vocabulary
            return Vocabulary
        return super().find_class(module, name)

def setup_vocab(path_to_vocab = 'store/vocab.pkl'):
    # vocab = pickle.load(open(path_to_vocab, 'rb'))
    vocab = CustomUnpickler(open(path_to_vocab, 'rb')).load()
    return vocab

def setup_model(model_type, vocab, hidden_dim = 2048):

    if model_type == 'avg_word_emb':
        encoder = AvgWordEmb()
        classifier = MLP(300*4)
    elif model_type == 'uni_lstm':
        encoder = UniLSTM(hidden_dim)
        classifier = MLP(hidden_dim*4)
    elif model_type == 'bi_lstm':
        encoder = BiLSTM(hidden_dim)
        classifier = MLP(hidden_dim*4*2)
    elif model_type == 'max_pool_lstm':
        encoder = MaxPoolLSTM(hidden_dim)
        classifier = MLP(hidden_dim*4*2)
    else:
        raise ValueError('Unknown model type')

    net = NLINet(encoder, classifier, vocab)
    model = Learner(net)

    return model, net



def find_checkpoint(ckpt_path, version):
    versions = os.listdir(os.path.join('logs', ckpt_path))
    version_ = versions[0] if len(versions) == 1 else version
    assert version == version_
    ckpts = os.listdir(os.path.join('logs', ckpt_path, version, 'checkpoints'))
    assert len(ckpts) == 1
    version_path = os.path.join('logs', ckpt_path, version)
    ckpt_path = os.path.join('logs', ckpt_path, version, 'checkpoints', ckpts[0])
    return ckpt_path, version_path

def load_model(model_type, path_to_vocab, ckpt_path, version):

    ckpt_path, _ = find_checkpoint(ckpt_path, version)
    logging.info(f'Loading model from {ckpt_path}')

    vocab = setup_vocab(path_to_vocab)
    _, net = setup_model(model_type, vocab)
    model = Learner.load_from_checkpoint(ckpt_path, net=net)
    model.eval()

    return model, vocab

def prep_sent(sent, vocab):
    return DataSetPadding(None, vocab).prepare_sent(sent)