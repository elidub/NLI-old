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
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import os
import json
import pytorch_lightning as pl
import torch
import argparse

from setup import load_model, prep_sent, find_checkpoint
from data import NLIDataModule


def parse_option():
    parser = argparse.ArgumentParser(description="Saving results of NLI and SentEval")

    parser.add_argument('--model_type', type=str, default='uni_lstm', help='Model type', choices=['avg_word_emb', 'uni_lstm', 'bi_lstm', 'max_pool_lstm'])
    parser.add_argument('--ckpt_path', type=str, default = None, help='Path to save checkpoint')
    parser.add_argument('--version', default= 'version_0', help='Version of the model to load')

    parser.add_argument('--tranfer_results', action='store_true', default = True, help='Get the transfer results')
    parser.add_argument('--nli_results',     action='store_true', default = True, help='Get the nli results')


    # parser.add_argument('--path_to_results', type=str, default='store/results.txt', help='Path to results')
    parser.add_argument('--path_to_vocab', type=str, default='store/vocab.pkl', help='Path to vocab')
    # parser.add_argument('--path_to_data', type=str, default = './SentEval/data', help='Path to data')

    parser.add_argument('--num_workers', type=int, default=3, help='Number of workers for dataloader')


    args = parser.parse_args()

    return args

class TransferResults:
    def __init__(self, args, tasks_with_acc_given = None):
        
        # Read the results
        _, version_path = find_checkpoint(args.ckpt_path, args.version)
        with open(os.path.join(version_path, 'results.txt'), 'r') as f:
        # with open(os.path.join('results.txt'), 'r') as f:
            results = json.load(f)
        self.results = results

        # Assert that the tasks with acc are the same as the ones given
        self.tasks_with_acc = self.get_tasks_with_acc(tasks_with_acc_given)

    def get_tasks_with_acc(self, tasks_with_acc_given):
        task_with_acc = {task for task in self.results if 'acc' in self.results[task]}
        if tasks_with_acc_given is not None:
            assert task_with_acc == tasks_with_acc_given, f'{task_with_acc} != {tasks_with_acc_given}'
        return task_with_acc

    def get_transfer_accs(self):
        dev_accs = {}
        num_dev_samples = {}
        for task, task_data in self.results.items():
            if task not in self.tasks_with_acc:
                continue
            dev_accs[task] = task_data['devacc']
            num_dev_samples[task] = task_data['ndev']

        # Calculate macro accuracy
        macro_acc = sum(dev_accs.values()) / len(dev_accs)

        # Calculate micro accuracy
        total_dev_samples = sum(num_dev_samples.values())
        micro_acc = sum(dev_accs[task] * num_dev_samples[task] / total_dev_samples for task in dev_accs)

        return {'micro': micro_acc, 'macro': macro_acc}
    
class NLIResults:
    def __init__(self, args):

        self.model, vocab = load_model(args.model_type, args.path_to_vocab, args.ckpt_path, args.version)
        self.datamodule = NLIDataModule(vocab=vocab, batch_size=64, num_workers=args.num_workers)
        self.trainer = pl.Trainer(
            logger = False,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        )

    def test(self):
        test_acc = self.trainer.test(self.model, datamodule=self.datamodule, verbose = False)[0]['test_acc']
        return test_acc
    
    def validate(self):
        val_acc = self.trainer.validate(self.model, datamodule=self.datamodule, verbose = False)[0]['val_acc']
        return val_acc
    
    def get_nli_accs(self):
        return {'val': self.validate()*100., 'test': self.test()*100., }
    
    def get_nli_preds(self):
        y_hat, y = self.trainer.predict(self.model, datamodule=self.datamodule)[0]
        y_pred = torch.nn.functional.softmax(y_hat, dim=1)
        return y_pred


def main(args):

    args.ckpt_path = args.ckpt_path if args.ckpt_path is not None else args.model_type

    _, version_path = find_checkpoint(args.ckpt_path, args.version)
    
    if args.tranfer_results:
        transfer_results = TransferResults(args, {'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC', 'SICKEntailment'})
        transfer_accs = transfer_results.get_transfer_accs()
    else: 
        transfer_accs = {}

    if args.nli_results:
        nli_results = NLIResults(args)

        nli_accs = nli_results.get_nli_accs()

        nli_preds = nli_results.get_nli_preds()
        torch.save(nli_preds, os.path.join(version_path, 'preds.pt'))
        logging.info(f'Pred: {nli_preds}')
    else:
        nli_accs = {}

    accs = {**nli_accs, **transfer_accs}
    accs = {k: round(v, 1) for k, v in accs.items()}

    with open(os.path.join(version_path, 'accs.txt'), 'w') as f:
        json.dump(accs, f)

    logging.info(f'{args.model_type} : {accs}')



if __name__ == "__main__":
    args = parse_option()
    print(args)
    main(args)