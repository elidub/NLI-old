import torch
import pytorch_lightning as pl
import argparse

from data import NLIDataModule, Vocabulary
from models import AvgWordEmb, UniLSTM, BiLSTM, MaxPoolLSTM, MLP, NLINet
from learner import Learner

import pickle
from datasets import load_from_disk




def parse_option():
    parser = argparse.ArgumentParser(description="Training NLI models")

    parser.add_argument('--model', type=str, default='uni_lstm', help='Model type', choices=['avg_word_emb', 'uni_lstm', 'bi_lstm', 'max_pool_lstm'])
    parser.add_argument('--load_vocab', action='store_true', default = False, help='Load word vectors and vocabulary')
    
    parser.add_argument('--ckpt_path', type=str, help='Path to save checkpoint')

    parser.add_argument('--epochs', type=int, default=10, help='Max number of training epochs')
    parser.add_argument('--num_workers', type=int, default=3, help='Number of workers for dataloader')

    # parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    # parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    # parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    # parser.add_argument('--hidden_size', type=int, default=2048, help='hidden size')
    # parser.add_argument('--embed_size', type=int, default=300, help='embedding size')
    # parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
    # parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    # parser.add_argument('--bidirectional', type=bool, default=False, help='bidirectional')
    # parser.add_argument('--device', type=str, default='gpu', help='device')

    ## boolean parser arguments


    args = parser.parse_args()

    return args



def main(args):

    if args.load_vocab:
        print('Loading dataset and building vocabulary...')
        vocab = pickle.load(open('store/vocab.pkl', 'rb'))

    else:
        dataset_snli = load_from_disk('data/snli')
        samples = dataset_snli['train']['premise'] + dataset_snli['train']['hypothesis']
        vocab = Vocabulary(samples, path_to_vec = "data/glove.840B.300d.txt")

        with open('store/vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)


    hidden_dim = 2048

    if args.model == 'avg_word_emb':
        encoder = AvgWordEmb()
        classifier = MLP(300*4)
    elif args.model == 'uni_lstm':
        encoder = UniLSTM(hidden_dim)
        classifier = MLP(hidden_dim*4)
    elif args.model == 'bi_lstm':
        encoder = BiLSTM(hidden_dim)
        classifier = MLP(hidden_dim*4*2)
    elif args.model == 'max_pool_lstm':
        encoder = MaxPoolLSTM(hidden_dim)
        classifier = MLP(hidden_dim*4*2)
    else:
        raise ValueError('Unknown model type')


    net = NLINet(encoder, classifier, vocab)
    model = Learner(net)

    datamodule = NLIDataModule(vocab = vocab, batch_size=64, num_workers=args.num_workers)

    trainer = pl.Trainer(
        default_root_dir = args.ckpt_path,
        max_epochs = args.epochs, 
        log_every_n_steps = 1, 
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
        callbacks = [pl.callbacks.ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
    )
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    args = parse_option()
    print(args)
    main(args)