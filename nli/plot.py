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
from torch import utils

import matplotlib.pyplot as plt

from setup import find_checkpoint


class PlotResults:
    

    def __init__(self, models, versions):

        assert len(models) == len(versions), 'models and versions must be of the same length'

        models_all = {'avg_word_emb': 'Avg WordEmb', 'uni_lstm': 'Uni-LSTM', 'bi_lstm': 'Bi-LSTM', 'max_pool_lstm': 'BiLSTM-Max'}
        self.models = {k: v for k, v in models_all.items() if k in models}

        assert len(set(versions)) == 1, '#TODO: fix self.version to be compatible with different versions for different models'
        self.version = versions[0]

        self.colors = {'Entailment' : 'tab:green', 'Neutral' : 'tab:blue', 'Contradiction' : 'tab:red'}

    def plot_examples(self):

        with open('data/examples_snli.json', 'r') as f:
            predict_data = json.load(f)
        predict_data

        size = 2.5
        label_dict = {0 : 1/6, 1 : 0.5, 2 : 5/6}

        n_pred = len(predict_data['label'])
        n_models = len(self.models)
        labels = list(self.colors.keys())

        fig, axs = plt.subplots(n_models, n_pred, figsize=(n_pred*size, n_models*size), sharey=True, sharex=True)

        x = np.arange(3)
        for row_i, model in enumerate(self.models.keys()):
            _, version_path = find_checkpoint(model, self.version)
            preds =  torch.load(os.path.join(version_path, 'store/example_preds.pt'))
            for col_i, (pred, ax) in enumerate(zip(preds, axs[row_i])):
                ax.bar(x, pred, color = list(self.colors.values()))
                ax.set_xticks([])
                ax.set_yticks([0, 0.5, 1])
                ax.set_ylim(0, 1)
                label = label_dict[predict_data['label'][col_i]]

                if col_i == 0:
                    ax.set_ylabel(self.models[model], size = 12)
                if row_i == 0:
                    ax.set_title(f"{predict_data['premise'][col_i]};\n{predict_data['hypothesis'][col_i]}", size = 9, style = 'italic')
                if row_i == n_models-1:
                    ax.annotate('', xy=(label, -0.), xycoords='axes fraction', xytext=(label, -0.2), 
                            arrowprops=dict(arrowstyle="->", color='black', lw=3))

        handles = [plt.Rectangle((0,0),1,1, color=self.colors[label]) for label in labels]
        fig.legend(handles, labels, loc='lower center', prop={'size': 12}, ncol=3, bbox_to_anchor=(0.5, -0.02))

        fig.supylabel('Prediction probability')
        fig.supxlabel('Predicted class', y = -0.05)
        fig.suptitle('Premise; Hypothesis', style = 'italic', y = 0.95)


        plt.tight_layout()

        return fig



    def plot_violin(self):
        size = 2.5

        n_class = 3
        n_models = len(self.models)
        labels = list(self.colors.keys())

        fig, axs = plt.subplots(n_models, n_class, figsize=(n_class*size, n_models*size), sharey=True, sharex=True)

        x = np.arange(n_class)
        for row_i, model in enumerate(self.models.keys()):
            _, version_path = find_checkpoint(model, self.version)
            preds =  torch.load(os.path.join(version_path, 'store/test_preds.pt'))
            trues =  torch.load(os.path.join(version_path, 'store/test_trues.pt'))


            # device preds into a dictionary based upon the value of true
            pred_dict = {true.item() : torch.tensor([]) for true in trues.unique()}
            for pred, true in zip(preds, trues):
                pred_dict[true.item()] = torch.cat((pred_dict[true.item()], pred.unsqueeze(0)), dim = 0)

            for col_i, ((label, pred), ax) in enumerate(zip(pred_dict.items(), axs[row_i])):

                r = ax.violinplot(pred.T, positions=x, showmeans=False, showmedians=False, showextrema=False)

                for pc, color in zip(r['bodies'], self.colors.values()):
                    pc.set_facecolor(color)
                    pc.set_alpha(1)

                ax.set_xticks([])
                ax.set_yticks([0, 0.5, 1])
                ax.set_ylim(0, 1)

                if col_i == 0:
                    ax.set_ylabel(self.models[model], size = 12)
                if row_i == 0:
                    ax.set_title(labels[col_i], size = 12, color = self.colors[labels[col_i]], fontweight = 'bold')

        handles = [plt.Rectangle((0,0),1,1, color=self.colors[label]) for label in labels]
        fig.legend(handles, labels, loc='lower center', prop={'size': 12}, ncol=3, bbox_to_anchor=(0.5, -0.02))

        fig.supylabel('Prediction probability')
        fig.supxlabel('Predicted class', y = -0.05)
        fig.suptitle('True class', fontweight = 'bold', y = 0.95)


        plt.tight_layout()

        return fig
    
    def __call__(self, fig_dir = 'figs'):
        fig = self.plot_examples().plot()
        fig.savefig('figs/examples.png', dpi = 300)

        fig = self.plot_violin().plot()
        fig.savefig('figs/violin.png', dpi = 300)
        