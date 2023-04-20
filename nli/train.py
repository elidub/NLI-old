import torch
import pytorch_lightning as pl
import argparse
import pickle

from data import NLIDataModule
from setup import setup_vocab, setup_model

# from models import AvgWordEmb, UniLSTM, BiLSTM, MaxPoolLSTM, MLP, NLINet
# from learner import Learner
# from datasets import load_from_disk




def parse_option():
    parser = argparse.ArgumentParser(description="Training NLI models")

    parser.add_argument('--model_type', type=str, default='uni_lstm', help='Model type', choices=['avg_word_emb', 'avg_word_emb2', 'uni_lstm', 'bi_lstm', 'max_pool_lstm'])
    parser.add_argument('--feature_type', type=str, default = 'baseline', help='Type of features to use', choices=['baseline', 'multiplication', 'exponent'])
    
    parser.add_argument('--ckpt_path', type=str, default = None, help='Path to save checkpoint')
    parser.add_argument('--version', default=None, help='Version of the model to load')

    parser.add_argument('--path_to_vocab', type=str, default='store/vocab.pkl', help='Path to vocab')

    parser.add_argument('--epochs', type=int, default=20, help='Max number of training epochs')
    parser.add_argument('--num_workers', type=int, default=3, help='Number of workers for dataloader')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')

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

    ckpt_path = args.ckpt_path if args.ckpt_path is not None else args.model_type
    pl.seed_everything(args.seed, workers=True)

    vocab    = setup_vocab(args.path_to_vocab)
    model, _ = setup_model(args.model_type, vocab, args.feature_type)
    datamodule = NLIDataModule(vocab=vocab, batch_size=64, num_workers=args.num_workers)

    trainer = pl.Trainer(
        logger = pl.loggers.TensorBoardLogger('logs', name=ckpt_path, version=args.version),
        max_epochs = args.epochs, 
        log_every_n_steps = 1, 
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
        callbacks = [pl.callbacks.ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        deterministic = True,
    )
    trainer.fit(model,  datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    args = parse_option()
    print(args)
    main(args)