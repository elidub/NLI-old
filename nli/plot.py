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
import pandas as pd

from setup import find_checkpoint
from results import NewSentence

import numpy as np

from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tqdm import tqdm


class PlotResults:
    

    def __init__(self, models, versions, dims):

        assert len(models) == len(versions) == len(dims), 'models and versions must be of the same length'

        models_all = {'avg_word_emb': 'AvgWordEmb', 'uni_lstm': 'UniLSTM', 'bi_lstm': 'BiLSTM-last', 'max_pool_lstm': 'BiLSTM-Max'}
        self.models = {k: v for k, v in models_all.items() if k in models}

        assert len(set(versions)) == 1, '#TODO: fix self.version to be compatible with different versions for different models'
        self.version = versions[0]

        self.dims = dims

        self.colors = {'Entailment' : 'tab:green', 'Neutral' : 'tab:blue', 'Contradiction' : 'tab:red'}

    def plot_examples(self):

        with open('data/examples_snli.json', 'r') as f:
            predict_data = json.load(f)

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
        fig.suptitle('Premise; Hypothesis', style = 'italic', y = 1)


        plt.tight_layout()

        fig.show()

    
    def plot_new_sample(self, premise, hypothesis, model_type, feature_type = 'baseline'):

        pred = NewSentence(premise, hypothesis, model_type, feature_type = feature_type).pred


        x = np.arange(3)
        size = 3
        ratio = 3
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(ratio/2*size, size), sharey=True, sharex=True, gridspec_kw={'width_ratios': [ratio, 1]})
        ax.bar(x, pred, color = list(self.colors.values()))
        ax.set_xticks([])
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylim(0, 1)


        labels = list(self.colors.keys())

        # handles = [plt.Rectangle((0,0),1,1, color=self.colors[label]) for label in labels]
        # ax.legend(handles, labels, loc = 'upper left', prop={'size': 12})

        # plot labels on top of bars
        for i, (v, label) in enumerate(zip(pred, labels)):
            # ax.text(i - 0.1, v + 0.05, label, color=self.colors[label], rotation = 90, fontweight='bold')
            ax.text(i - 0.1, 0.05, label, color='black', rotation = 90, fontweight='bold')
            


        # fig.suptitle(self.models[model], size = 12)
        ax.set_xlabel('Predicted class', y = -0.05)
        ax.set_ylabel('Prediction probability')

        ax2.text(0., 0.5, f"Model:\n{self.models[model_type]}\n\nPremise:\n{premise}\n\nHypothesis:\n{hypothesis}", size = 9,
                 ha = 'left', va = 'center', wrap = True)
        ax2.axis('off')

        # fig.legend(handles, labels, loc='lower center', prop={'size': 12}, ncol=3, bbox_to_anchor=(0.5, -0.02))

        # fig.suptitle('Premise; Hypothesis', style = 'italic', y = 0.95)

        plt.tight_layout()

        fig.show()



    def plot_embeddings(self):
        size = 2.5

        n_models = len(self.models)
        labels = list(self.colors.keys())

        c = 'tab:blue'

        fig, axs = plt.subplots(1, n_models, figsize=(n_models*size, size))

        for ax, model in zip(axs, self.models.keys()):
            _, version_path = find_checkpoint(model, self.version)
            emb =  torch.load(os.path.join(version_path, 'store/test_emb.pt'))
        
            mean, std = torch.mean(emb, dim=0), torch.std(emb, dim=0)

            ax.plot(mean, color = c)
            ax.fill_between(range(emb.shape[1]), mean - std, mean + std, alpha=0.3, color=c)
            ax.set_title(self.models[model], size = 12)

        fig.supxlabel('Word embedding')

        plt.tight_layout()

        fig.show()



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
        fig.suptitle('True class', fontweight = 'bold', y =1)


        plt.tight_layout()

        fig.show()

    def plot_bars(self):
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

                bins = np.arange(-0.5, 3.5, 1.)
                heigths, _ = np.histogram(torch.argmax(pred, dim = 1).numpy() , bins = bins, density = True)
                ax.bar(np.arange(3), heigths, color = self.colors.values()) 

                ax.set_xticks([])
                ax.set_yticks([0, 0.5, 1])
                ax.set_ylim(0, 1)

                if col_i == 0:
                    ax.set_ylabel(self.models[model], size = 12)
                if row_i == 0:
                    ax.set_title(labels[col_i], size = 12, color = self.colors[labels[col_i]], fontweight = 'bold')

        handles = [plt.Rectangle((0,0),1,1, color=self.colors[label]) for label in labels]
        fig.legend(handles, labels, loc='lower center', prop={'size': 12}, ncol=3, bbox_to_anchor=(0.5, -0.02))

        fig.supylabel('Total predictions (%)')
        fig.supxlabel('Predicted class', y = -0.05)
        fig.suptitle('True class', fontweight = 'bold', y = 1)


        plt.tight_layout()

        fig.show()
    


    def print_results(self):

        acc_dict = {}

        for model, dim in zip(self.models, self.dims):
            _, version_path = find_checkpoint(model, self.version)

            # read file from  os.path.join(version_path, 'store/accs.txt' with json
            with open(os.path.join(version_path, 'store/accs.txt'), 'r') as f:
                accs = json.load(f)

            acc_dict[model] = accs
            acc_dict[model]['dim'] = dim

        dtypes = {'val': float, 'test': float, 'micro': float, 'macro': float, 'dim': int}
        alt_titles = {'val': 'dev', 'test': 'test', 'micro': 'micro', 'macro': 'macro'}
        models_all =  {'avg_word_emb': 'Avg WordEmb', 'uni_lstm': 'Uni-LSTM', 'bi_lstm': 'Bi-LSTM', 'max_pool_lstm': 'BiLSTM-Max'}
        custom_order = ['dim', 'val', 'test', 'micro', 'macro']

        df = pd.DataFrame(acc_dict).T.astype(dtypes).loc[:, custom_order].rename(columns=alt_titles, index = self.models)

        return df
    
    def __call__(self, fig_dir = 'figs'):
        fig = self.plot_examples().plot()
        fig.savefig('figs/examples.png', dpi = 300)

        fig = self.plot_violin().plot()
        fig.savefig('figs/violin.png', dpi = 300)
        

class PlotResultsMult:
    def __init__(self, models, features_types, versions,):


        models_all = {'avg_word_emb': 'AvgWordEmb', 'uni_lstm': 'UniLSTM', 'bi_lstm': 'BiLSTM-last', 'max_pool_lstm': 'BiLSTM-Max'}
        self.models = {k: v for k, v in models_all.items() if k in models}

        self.features_types = features_types

        assert len(set(versions)) == 1, '#TODO: fix self.version to be compatible with different versions for different models'
        self.version = versions[0]

        return

        assert len(models) == len(ckpt_paths) == len(versions) == len(dims), 'models and versions must be of the same length'

        models_all = {'avg_word_emb': 'AvgWordEmb', 'uni_lstm': 'UniLSTM', 'bi_lstm': 'BiLSTM-last', 'max_pool_lstm': 'BiLSTM-Max'}
        self.models = {k: v for k, v in models_all.items() if k in models}

        ckpt_paths_all = {'avg_word_emb': 'AvgWordEmb', 'uni_lstm': 'UniLSTM', 'bi_lstm': 'BiLSTM-last', 'max_pool_lstm': 'BiLSTM-Max',
                          'avg_word_emb_mult': 'AvgWordEmb (Mult)', 'uni_lstm_mult': 'UniLSTM (Mult)', 'bi_lstm_mult': 'BiLSTM-last (Mult)', 'max_pool_lstm_mult': 'BiLSTM-Max (Mult)'}
        self.ckpt_paths = {k: v for k, v in ckpt_paths_all.items() if k in ckpt_paths}

        assert len(set(versions)) == 1, '#TODO: fix self.version to be compatible with different versions for different models'
        self.version = versions[0]

        self.dims = dims

        self.colors = {'Entailment' : 'tab:green', 'Neutral' : 'tab:blue', 'Contradiction' : 'tab:red'}

    def print_results(self):


        
        
        
        features_types = {'baseline': '', 'multiplication': '_mult'}

        version = 'version_0'

        acc_dict = {}


        for model in self.models:
            for features_type in self.features_types:
                _, version_path = find_checkpoint(model + features_types[features_type], version)

                # read file from  os.path.join(version_path, 'store/accs.txt' with json
                with open(os.path.join(version_path, 'store/accs.txt'), 'r') as f:
                    accs = json.load(f)
                    accs = {k+features_types[features_type]: v for k, v in accs.items()}

                # check if model is in acc_dict
                if model not in acc_dict:
                    acc_dict[model] = accs
                else:
                    acc_dict[model].update(accs)

        dtypes = {'val': float, 'test': float, 'val_mult': float, 'test_mult': float,}
        alt_titles = {'val': 'dev', 'test': 'test', 'val_mult': 'dev (mult)', 'test_mult': 'test (mult)', }
        models_all =  {'avg_word_emb': 'Avg WordEmb', 'uni_lstm': 'Uni-LSTM', 'bi_lstm': 'Bi-LSTM', 'max_pool_lstm': 'BiLSTM-Max'}
        custom_order = ['val', 'val_mult', 'test', 'test_mult']

        df = pd.DataFrame(acc_dict).T.astype(dtypes).loc[:, custom_order].rename(columns=alt_titles, index = self.models)
        return df

    def plot_multipliers(self, calculate = False):
        mults_dict = {}

        for model in self.models:
            model_feature = model + '_mult'
            _, version_path = find_checkpoint(model_feature, self.version)

            if calculate:

                event_acc = EventAccumulator(version_path, size_guidance={"scalars": 0})
                event_acc.Reload()

                scalar_tags = event_acc.Tags()["scalars"]

                mults = []
                for i in tqdm(range(4)):
                    scalar_events = event_acc.Scalars(f'multiplier_{i}')
                    mult = [s.value for s in scalar_events]
                    mults.append(mult)
                mults = np.array(mults)

                # save mults at version_path
                np.save(os.path.join(version_path, 'store/mults.npy'), mults)

            else:
                # load mults at version_path
                mults = np.load(os.path.join(version_path, 'store/mults.npy'))
            
            mults_dict[model_feature] = mults


        mult_labels = ['$u$', '$v$', '$u \cdot v$', '$\mid u - v \mid$']

        size = 3
        n_models = len(mults_dict)
        fig, axs = plt.subplots(1, n_models, figsize=(n_models*size, size), sharey=True)

        for mult, model, ax in zip(mults_dict.values(), list(self.models.values()), axs):

            ax.set_title(model)

            for i, mult_label in enumerate(mult_labels):
                ax.plot(mult[i], label=mult_label)

            # plot a horizontal line at 1
            ax.plot(mult[0] / mult[1], label='$u/v$ (calc.)', linestyle='--', zorder  = 0)
            ax.axhline(1, color='black', linestyle='--', linewidth=0.5)

        axs[0].legend(loc = 'upper left')

        fig.supxlabel('Iterations')
        fig.supylabel('Multiplier value')
        fig.tight_layout()

        plt.show()