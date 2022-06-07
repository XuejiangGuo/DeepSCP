# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : wangbing
# @FILE     : DeepSCP.py
# @Time     : 9/20/2021 9:29 PM
# @Desc     : DeepSCP: utilizing deep learning to boost single-cell proteome coverage


import numpy as np
import pandas as pd
import scipy as sp
import lightgbm as lgb
import networkx as nx
import matplotlib.pyplot as plt

from copy import deepcopy
from time import time
from joblib import Parallel, delayed
from scipy.stats.mstats import gmean
from bayes_opt import BayesianOptimization
from triqler.qvality import getQvaluesFromScores

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = 'Arial'


def showcols(df):
    cols = df.columns.tolist()
    cols = cols + (10 - len(cols) % 10) % 10 * ['None']
    cols = np.array(cols).reshape(-1, 10)
    return pd.DataFrame(data=cols, columns=range(1, 11))


def Prob2PEP(y, y_prob):
    label = np.array(deepcopy(y))
    score = np.array(deepcopy(y_prob))
    srt_idx = np.argsort(-score)
    label_srt = label[srt_idx]
    score_srt = score[srt_idx]
    targets = score_srt[label_srt == 1]
    decoys = score_srt[label_srt != 1]
    _, pep = getQvaluesFromScores(targets, decoys, includeDecoys=True)
    return pep[np.argsort(srt_idx)]


def Score2Qval(y0, y_score):
    y = np.array(deepcopy(y0)).flatten()
    y_score = np.array(y_score).flatten()
    y[y != 1] = 0
    srt_idx = np.argsort(-y_score)
    y_srt = y[srt_idx]
    cum_targets = y_srt.cumsum()
    cum_decoys = np.abs(y_srt - 1).cumsum()
    FDR = np.divide(cum_decoys, cum_targets)
    qvalue = np.zeros(len(FDR))
    qvalue[-1] = FDR[-1]
    qvalue[0] = FDR[0]
    for i in range(len(FDR) - 2, 0, -1):
        qvalue[i] = min(FDR[i], qvalue[i + 1])
    qvalue[qvalue > 1] = 1
    return qvalue[np.argsort(srt_idx)]


def GroupProteinPEP2Qval(data, file_column, protein_column, target_column, pep_column):
    data['Protein_label'] = data[protein_column] + '_' + data[target_column].map(str)
    df = []
    for i, j in data.groupby(file_column):
        df_pro = j[['Protein_label', target_column, pep_column]].sort_values(pep_column).drop_duplicates(
                        subset='Protein_label', keep='first')
        df_pro['protein_qvalue'] = Score2Qval(df_pro[target_column].values, -df_pro[pep_column].values)
        df.append(pd.merge(j,
                        df_pro[['Protein_label', 'protein_qvalue']],
                        on='Protein_label',
                        how='left'))
    return pd.concat(df, axis=0).drop('Protein_label', axis=1)


class SampleRT:
    def __init__(self, r=3):
        self.r = r

    def fit_tranform(self, data, file_column, peptide_column, RT_column, score_column,target_column):
        self.file_column = file_column
        self.peptide_column = peptide_column
        self.RT_column = RT_column
        self.score_column = score_column
        self.target_column = target_column

        data['File_Pep'] = data[self.file_column] + '+' + data[self.peptide_column]
        dftagraw = data[data[self.target_column] == 1]
        dfrevraw = data[data[self.target_column] != 1]

        dftag1 = self.makePeptied_File(self.Repeat3(dftagraw))
        dfrev1 =self.makePeptied_File(self.Repeat3(dfrevraw))

        dftag2 = self.makeRTfeature(dftag1)
        dfrev2 = self.makeRTfeature(dfrev1)

        reg = ElasticNet()
        pred_tag_tag = pd.DataFrame(data=np.zeros_like(dftag1.values), columns=dftag1.columns, index=dftag1.index)
        pred_rev_tag = pd.DataFrame(data=np.zeros_like(dfrev1.values), columns=dfrev1.columns, index=dfrev1.index)
        pred_rev_rev = pd.DataFrame(data=np.zeros_like(dfrev1.values), columns=dfrev1.columns, index=dfrev1.index)
        pred_tag_rev = pd.DataFrame(data=np.zeros_like(dftag1.values), columns=dftag1.columns, index=dftag1.index)

        scores_tag_tag = []
        scores_rev_tag = []
        scores_tag_rev = []
        scores_rev_rev = []

        samples = dftag1.columns.tolist()
        for sample in samples:
            y_tag = dftag1[~dftag1[sample].isna()][sample]
            X_tag = dftag2.loc[dftag2.index.isin(y_tag.index)]

            y_rev = dfrev1[~dfrev1[sample].isna()][sample]
            X_rev = dfrev2.loc[dfrev2.index.isin(y_rev.index)]

            reg_tag = reg
            reg_tag.fit(X_tag, y_tag)

            scores_rev_tag.append(reg_tag.score(X_rev, y_rev))
            scores_tag_tag.append(reg_tag.score(X_tag, y_tag))

            pred_rev_tag.loc[dfrev2.index.isin(y_rev.index), sample] = reg_tag.predict(X_rev)
            pred_tag_tag.loc[dftag2.index.isin(y_tag.index), sample] = reg_tag.predict(X_tag)

            reg_rev = reg
            reg_rev.fit(X_rev, y_rev)

            scores_rev_rev.append(reg_rev.score(X_rev, y_rev))
            scores_tag_rev.append(reg_rev.score(X_tag, y_tag))

            pred_rev_rev.loc[dfrev2.index.isin(y_rev.index), sample] = reg_rev.predict(X_rev)
            pred_tag_rev.loc[dftag2.index.isin(y_tag.index), sample] = reg_rev.predict(X_tag)

            pred_rev_tag[pred_rev_tag == 0.0] = np.nan
            pred_tag_tag[pred_tag_tag == 0.0] = np.nan
            pred_rev_rev[pred_rev_rev == 0.0] = np.nan
            pred_tag_rev[pred_tag_rev == 0.0] = np.nan

        self.cmp_scores = pd.DataFrame({'score': scores_tag_tag + scores_rev_tag + scores_tag_rev + scores_rev_rev,
                                   'type': ['RT(tag|tag)'] * len(scores_tag_tag) + ['RT(rev|tag)'] * len(scores_rev_tag) +
                                           ['RT(tag|rev)'] * len(scores_tag_rev) + ['RT(rev|rev)'] * len(scores_tag_rev)})

        pred_rev = pd.merge(self.makeRTpred(pred_rev_rev, 'RT(*|rev)'),
                            self.makeRTpred(pred_rev_tag, 'RT(*|tag)'), on='File_Pep')
        dfrevraw = pd.merge(dfrevraw, pred_rev, on='File_Pep', how='left')

        pred_tag = pd.merge(self.makeRTpred(pred_tag_rev, 'RT(*|rev)'),
                            self.makeRTpred(pred_tag_tag, 'RT(*|tag)'), on='File_Pep')
        dftagraw = pd.merge(dftagraw, pred_tag, on='File_Pep', how='left')

        df = pd.concat([dftagraw, dfrevraw], axis=0)
        df['DeltaRT'] = ((df[self.RT_column] - df['RT(*|rev)']).apply(abs) +
                               (df[self.RT_column] - df['RT(*|tag)']).apply(abs))
        df['DeltaRT'] = df['DeltaRT'].apply(lambda x: np.log2(x + 1))
        return df

    def Repeat3(self, data):
        df = pd.Series([i.split('+')[1] for i in data['File_Pep'].unique()]).value_counts()
        return data[data[self.peptide_column].isin(df[df >= self.r].index)]

    def makePeptied_File(self, data):
        data1 = data.sort_values(self.score_column, ascending=False).drop_duplicates(subset='File_Pep',
                                                                           keep='first')[
            [self.file_column, self.peptide_column, self.RT_column]]
        temp1 = list(zip(data1.iloc[:, 0], data1.iloc[:, 1], data1.iloc[:, 2]))
        G = nx.Graph()
        G.add_weighted_edges_from(temp1)
        df0 = nx.to_pandas_adjacency(G)
        df = df0[df0.index.isin(data1[self.peptide_column].unique())][data1[self.file_column].unique()]
        df.index.name = self.peptide_column
        df[df == 0.0] = np.nan
        return df

    def makeRTfeature(self, data):
        df_median = data.median(1)
        df_gmean = data.apply(lambda x: gmean(x.values[~np.isnan(x.values)]), axis=1)
        df_mean = data.mean(1)
        df_std = data.std(1)
        df_cv = df_std / df_mean * 100
        df_skew = data.apply(lambda x: x.skew(), axis=1)
        df = pd.concat([df_median, df_gmean, df_mean, df_std, df_cv, df_skew], axis=1)
        df.columns = ['Median', 'Gmean', 'Mean', 'Std', 'CV', 'Skew']
        return df

    def makeRTpred(self, data, name):
        m, n = data.shape
        cols = np.array(data.columns)
        inds = np.array(data.index)
        df_index = np.tile(inds.reshape(-1, 1), n).flatten()
        df_columns = np.tile(cols.reshape(1, -1), m).flatten()
        values = data.values.flatten()
        return pd.DataFrame({'File_Pep': df_columns + '+' + df_index, name: values})


class MQ_SampleRT:
    def __init__(self, r=3, filter_PEP=False):
        self.r = r
        self.filter_PEP = filter_PEP

    def fit_tranform(self, data):
        if self.filter_PEP:
            data = data[data['PEP'] <= self.filter_PEP]
        dftagraw = self.get_tag(data)
        dfrevraw = self.get_rev(data)
        df = pd.concat([self.get_tag(data), self.get_rev(data)], axis=0)
        sampleRT = SampleRT(r=self.r)
        dfdb = sampleRT.fit_tranform(df,
                                     file_column='Experiment',
                                     peptide_column='Modified sequence',
                                     RT_column='Retention time',
                                     score_column='Score',
                                     target_column='label')
        self.cmp_scores = sampleRT.cmp_scores
        dfdb['PEPRT'] = dfdb['DeltaRT'] * (1 + dfdb['PEP'])
        dfdb['ScoreRT'] = dfdb['Score'] / (1 + dfdb['DeltaRT'])
        return dfdb

    def get_tag(self, df):
        df = df[~df['Proteins'].isna()]
        df_tag = df[(~df['Proteins'].str.contains('CON_')) &
                    (~df['Leading razor protein'].str.contains('REV__'))]
        df_tag['label'] = 1
        return df_tag

    def get_rev(self, df):
        df_rev = df[(df['Leading razor protein'].str.contains('REV__')) &
                    (~df['Leading razor protein'].str.contains('CON__'))]
        df_rev['label'] = 0
        return df_rev


class IonCoding:
    def __init__(self, bs=1000, n_jobs=-1):
        ino = [
            '{0}{2};{1}{2};{0}{2}(2+);{1}{2}(2+);{0}{2}-NH3;{1}{2}-NH3;{0}{2}(2+)-NH3;{1}{2}(2+)-NH3;{0}{2}-H20;{1}{2}-H20;{0}{2}(2+)-H20;{1}{2}(2+)-H20'.
                format('b', 'y', i) for i in range(1, 47)]
        ino = np.array([i.split(';') for i in ino]).flatten()
        self.MI0 = pd.DataFrame({'MT': ino})
        self.bs = bs
        self.n_jobs = n_jobs

    def fit_transfrom(self, data):
        print('++++++++++++++++OneHotEncoder CMS(Chage + Modified sequence)++++++++++++++')
        t0 = time()
        x = self.onehotfeature(data['CMS']).reshape(data.shape[0], -1, 30)
        print('using time', time() - t0)
        print('x shape: ', x.shape)
        print('++++++++++++++++++++++++++Construct Ion Intensities Array++++++++++++++++++++')
        t0 = time()
        y = self.ParallelinoIY(data[['Matches', 'Intensities']])
        print('using time', time() - t0)
        print('y shape: ', y.shape)
        return x, y

    def onehotfeature(self, df0, s=48):
        df = df0.apply(lambda x: x + (s - len(x)) * 'Z')
        # B: '_(ac)M(ox)'; 'J': '_(ac)'; 'O': 'M(ox)'; 'Z': None
        aminos = '123456ABCDEFGHIJKLMNOPQRSTVWYZ'
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(np.repeat(np.array(list(aminos)), s).reshape(-1, s))
        seqs = np.array(list(df.apply(list)))
        return enc.transform(seqs).toarray().reshape(df.shape[0], -1, 30)

    def ParallelinoIY(self, data):
        datasep = [data.iloc[i * self.bs: (i + 1) * self.bs] for i in range(data.shape[0] // self.bs + 1)]
        paraY = Parallel(n_jobs=self.n_jobs)(delayed(self.dataapp)(i) for i in datasep)
        return np.vstack(paraY).reshape(data.shape[0], -1, 12)

    def inoIY(self, x):
        MI = pd.DataFrame({'MT0': x[0].split(';'),  'IY': [float(i) for i in x[1].split(';')]})
        dk = pd.merge(self.MI0, MI, left_on='MT', right_on='MT0', how='left').drop('MT0', axis=1)
        dk.loc[dk.IY.isna(), 'IY'] = 0
        dk['IY'] = dk['IY'] / dk['IY'].max()
        return dk['IY'].values

    def dataapp(self, data):
        return np.array(list(data.apply(self.inoIY, axis=1)))


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(CNN_BiLSTM, self).__init__()
        self.cov = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,
                             out_channels=64,
                             kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.lstm = nn.LSTM(input_size=64,
                            hidden_size=hidden_dim,
                            num_layers=layer_dim,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.5)

        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, output_dim),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.cov(x.permute(0, 2, 1))
        l_out, (l_hn, l_cn) = self.lstm(x.permute(0, 2, 1), None)
        x = self.fc(l_out)
        return x


class DeepSpec:
    def __init__(self, model=None, seed=0, test_size=0.2, lr=1e-3, l2=0.0,
                 batch_size=1024, epochs=1000, nepoch=50, patience=50,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.test_size = test_size
        self.seed = seed
        self.batch_size = batch_size
        self.device = device
        self.patience = patience
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.nepoch = nepoch
        self.model = model

    def fit(self, bkmsms):
        print('+++++++++++++++++++++++++++Loading Trainset+++++++++++++++++++++')
        bkmsms['CMS'] = bkmsms['Charge'].map(str) + bkmsms['Modified sequence']
        bkmsms['CMS'] = bkmsms['CMS'].apply(lambda x: x.replace('_(ac)M(ox)', 'B').replace(
            '_(ac)', 'J').replace('M(ox)', 'O').replace('_', ''))
        bkmsms1 = self.selectBestmsms(bkmsms, s=100)[['CMS', 'Matches', 'Intensities']]
        x, y = IonCoding().fit_transfrom(bkmsms1)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.test_size, random_state=self.seed)
        y_true = np.array(y_val).reshape(y_val.shape[0], -1).tolist()
        train_db = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_db,
                                  batch_size=self.batch_size,
                                  num_workers=0,
                                  shuffle=True)

        val_db = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_db,
                                batch_size=self.batch_size,
                                num_workers=0,
                                shuffle=False)

        if self.model is None:
            torch.manual_seed(self.seed)
            self.model = CNN_BiLSTM(30, 256, 2, 12)
        model = self.model.to(self.device)

        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.l2)

        val_losses = []
        val_cosines = []
        self.val_cosine_best = 0.0
        counter = 0

        print('+++++++++++++++++++DeepSpec Training+++++++++++++++++++++')
        for epoch in range(1, self.epochs + 1):
            for i, (x_batch, y_batch) in enumerate(train_loader):
                model.train()
                batch_x = x_batch.to(self.device)
                batch_y = y_batch.to(self.device)
                out = model(batch_x)
                loss = loss_func(out, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = 0
                y_valpreds = []
                for a, b in val_loader:
                    val_x = a.to(self.device)
                    val_y = b.to(self.device)
                    y_valpred = model(val_x)
                    y_valpreds.append(y_valpred)
                    val_loss += loss_func(y_valpred, val_y).item() / len(val_loader)

                val_losses.append(val_loss)
                y_valpreds = torch.cat([y_vp for y_vp in y_valpreds], dim=0)
                y_pred = np.array(y_valpreds.cpu()).reshape(y_val.shape[0], -1).tolist()

                val_cosine = self.cosine_similarity(y_true, y_pred)
                val_cosines.append(val_cosine)

                if val_cosine.mean() >= self.val_cosine_best:
                    counter = 0
                    self.val_cosine_best = val_cosine.mean()
                    self.val_loss_best = val_loss
                    self.bestepoch = epoch
                    torch.save(model, 'DeepSpec.pkl')
                else:
                    counter += 1

                if epoch % self.nepoch == 0 or epoch == self.epochs:
                    print(
                        '[{}|{}] val_loss: {} | val_cosine: {}'.format(epoch, self.epochs, val_loss, val_cosine.mean()))

                if counter >= self.patience:
                    print('EarlyStopping counter: {}'.format(counter))
                    break

        print('best epoch [{}|{}] val_ loss: {} | val_cosine: {}'.format(self.bestepoch, self.epochs,
                                                                         self.val_loss_best, self.val_cosine_best))
        self.traininfor = {'val_losses': val_losses, 'val_cosines': val_cosines}

    def predict(self, evidence, msms):
        dfdb = deepcopy(evidence)
        msms = deepcopy(msms).rename(columns={'id': 'Best MS/MS'})
        dfdb['CMS'] = dfdb['Charge'].map(str) + dfdb['Modified sequence']
        dfdb['CMS'] = dfdb['CMS'].apply(lambda x: x.replace('_(ac)M(ox)', 'B').replace(
            '_(ac)', 'J').replace('M(ox)', 'O').replace('_', ''))
        dfdb = pd.merge(dfdb, msms[['Best MS/MS', 'Matches', 'Intensities']], on='Best MS/MS', how='left')
        dfdb1 = deepcopy(dfdb[(~dfdb['Matches'].isna()) &
                              (~dfdb['Intensities'].isna()) &
                              (dfdb['Length'] <= 47) &
                              (dfdb['Charge'] <= 6)])[['id', 'CMS', 'Matches', 'Intensities']]
        print('after filter none Intensities data shape:', dfdb1.shape)
        print('+++++++++++++++++++Loading Testset+++++++++++++++++++++')
        x_test, y_test = IonCoding().fit_transfrom(dfdb1[['CMS', 'Matches', 'Intensities']])
        self.db_test = {'Data': dfdb1, 'x_test': x_test, 'y_test': y_test}
        x_test = torch.tensor(x_test, dtype=torch.float)
        test_loader = DataLoader(x_test,
                                            batch_size=self.batch_size,
                                            num_workers=0,
                                            shuffle=False)
        print('+++++++++++++++++++DeepSpec Testing+++++++++++++++++++++')
        y_testpreds = []
        model = torch.load('DeepSpec.pkl').to(self.device)

        model.eval()
        with torch.no_grad():
            for test_x in test_loader:
                test_x = test_x.to(self.device)
                y_testpreds.append(model(test_x))
            y_testpred = torch.cat(y_testpreds, dim=0)

        y_test = np.array(y_test).reshape(y_test.shape[0], -1)
        y_testpred = np.array(y_testpred.cpu())
        self.db_test['y_testpred'] = y_testpred
        y_testpred = y_testpred.reshape(y_test.shape[0], -1)
        CS = self.cosine_similarity(y_test, y_testpred)
        self.db_test['Cosine'] = CS
        output = pd.DataFrame({'id': dfdb1['id'].values,  'Cosine': CS})
        dfdb2 = pd.merge(dfdb, output, on='id', how='left')
        dfdb2['PEPCosine'] = dfdb2['Cosine'] / (1 + dfdb2['PEP'])
        dfdb2['ScoreCosine'] = dfdb2['Score'] / (1 + dfdb2['Cosine'])
        return dfdb2

    def cosine_similarity(self, y, y_pred):
        a, b = np.array(y), np.array(y_pred)
        res = np.array([[sum(a[i] * b[i]), np.sqrt(sum(a[i] * a[i]) * sum(b[i] * b[i]))]
                        for i in range(a.shape[0])])
        return np.divide(res[:, 0], res[:, 1])   # Cosine or DP
        # return 1 - 2 * np.arccos(np.divide(res[:, 0], res[:, 1])) / np.pi  # SA

    def selectBestmsms(self, df, lg=47, cg=6, s=100):
        return df[(df['Reverse'] != '+') & (~df['Matches'].isna()) &
                  (~df['Intensities'].isna()) & (df['Length'] <= lg) &
                  (df['Charge'] <= cg) & (df['Type'].isin(['MSMS', 'MULTI-MSMS']))
                  & (df['Score'] > s)].sort_values(
            'Score', ascending=False).drop_duplicates(
            subset='CMS', keep='first')[['CMS', 'Matches', 'Intensities']]

    def ValPlot(self):
        val_losses = self.traininfor['val_losses']
        val_cosines = self.traininfor['val_cosines']
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        lns1 = ax.plot(range(1, len(val_losses) + 1), val_losses,
                       color='orange', label='Val Loss={}'.format(round(self.val_loss_best, 5)))
        lns2 = ax.axvline(x=self.bestepoch, ls="--", c="b", label='kk')
        plt.xticks(size=15)
        plt.yticks(size=15)

        ax2 = ax.twinx()
        lns3 = ax2.plot(range(1, len(val_losses) + 1), [i.mean() for i in val_cosines],
                        color='red', label='Val Cosine={}'.format(round(self.val_cosine_best, 4)))

        lns = lns1 + lns3
        labs = [l.get_label() for l in lns]

        plt.yticks(size=15)
        ax.set_xlabel("Epoch", fontsize=18)
        ax.set_ylabel("Val Loss", fontsize=18)
        ax2.set_ylabel("Val Cosine", fontsize=18)
        ax.legend(lns, labs, loc=10, fontsize=15)
        plt.tight_layout()


class LGB_bayesianCV:
    def __init__(self, params_init=dict({}), n_splits=3, seed=0):
        self.n_splits = n_splits
        self.seed = seed
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'n_jobs': -1,
            'random_state': self.seed,
            'is_unbalance': True,
            'silent': True
        }
        self.params.update(params_init)

    def fit(self, x, y):
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        self.__x = np.array(x)
        self.__y = np.array(y)
        self.__lgb_bayesian()
        self.model = lgb.LGBMClassifier(**self.params)
        # self.cv_predprob = cross_val_predict(self.model, self.__x, self.__y,
        #                                      cv=self.skf,  method="predict_proba")[:, 1]
        self.model.fit(self.__x, self.__y)
        # self.feature_importance = dict(zip(self.model.feature_name_, self.model.feature_importances_))

    def predict(self, X):
        return self.model.predict(np.array(X))

    def predict_proba(self, X):
        return self.model.predict_proba(np.array(X))

    def __lgb_cv(self, n_estimators, learning_rate,
                 max_depth, num_leaves,
                 subsample, colsample_bytree,
                 min_split_gain, min_child_samples,
                 reg_alpha, reg_lambda):

        self.params.update({
            'n_estimators':int(n_estimators),  # 树个数 (100, 1000)
            'learning_rate': float(learning_rate),  # 学习率 (0.001, 0.3)
            'max_depth': int(max_depth),  # 树模型深度 (3, 15)
            'num_leaves': int(num_leaves),  # 最大树叶子节点数31, (2, 2^md) (5, 1000)
            'subsample': float(subsample),  # 样本比例 (0.3, 0.9)
            'colsample_bytree': float(colsample_bytree),  # 特征比例 (0.3, 0.9)
            'min_split_gain': float(min_split_gain),  # 切分的最小增益 (0, 0.5)
            'min_child_samples': int(min_child_samples),  # 最小数据叶子节点(5, 1000)
            'reg_alpha': float(reg_alpha),  # l1正则 (0, 10)
            'reg_lambda': float(reg_lambda),  # l2正则 (0, 10)
        })

        model = lgb.LGBMClassifier(**self.params)
        cv_score = cross_val_score(model, self.__x, self.__y, scoring="roc_auc", cv=self.skf).mean()
        return cv_score

    def __lgb_bayesian(self):
        lgb_bo = BayesianOptimization(self.__lgb_cv,
                                      {
                                          'n_estimators': (100, 1000),  # 树个数 (100, 1000)
                                          'learning_rate': (0.001, 0.3),  # 学习率 (0.001, 0.3)
                                          'max_depth': (3, 15),  # 树模型深度 (3, 15)
                                          'num_leaves': (5, 1000),  # 最大树叶子节点数31, (2, 2^md) (5, 1000)
                                          'subsample': (0.3, 0.9),  # 样本比例 (0.3, 0.9)
                                          'colsample_bytree': (0.3, 0.9),  # 特征比例
                                          'min_split_gain': (0, 0.5),  # 切分的最小增益 (0, 0.5)
                                          'min_child_samples': (5, 200),  # 最小数据叶子节点(5, 1000)
                                          'reg_alpha': (0, 10),  # l1正则 (0, 10)
                                          'reg_lambda': (0, 10),  # l2正则 (0, 10)
                                      },
                                      random_state=self.seed,
                                      verbose=0)
        lgb_bo.maximize()
        self.best_auc = lgb_bo.max['target']
        lgbbo_params = lgb_bo.max['params']
        lgbbo_params['n_estimators'] = int(lgbbo_params['n_estimators'])
        lgbbo_params['learning_rate'] = float(lgbbo_params['learning_rate'])
        lgbbo_params['max_depth'] = int(lgbbo_params['max_depth'])
        lgbbo_params['num_leaves'] = int(lgbbo_params['num_leaves'])
        lgbbo_params['subsample'] = float(lgbbo_params['subsample'])
        lgbbo_params['colsample_bytree'] = float(lgbbo_params['colsample_bytree'])
        lgbbo_params['min_split_gain'] = float(lgbbo_params['min_split_gain'])
        lgbbo_params['min_child_samples'] = int(lgbbo_params['min_child_samples'])
        lgbbo_params['reg_alpha'] = float(lgbbo_params['reg_alpha'])
        lgbbo_params['reg_lambda'] = float(lgbbo_params['reg_lambda'])
        self.params.update(lgbbo_params)


class LgbBayes:
    def __init__(self, out_cv=3, inner_cv=3, seed=0):
        self.out_cv = out_cv
        self.inner_cv = inner_cv
        self.seed = seed

    def fit_tranform(self, data, feature_columns, target_column, file_column, protein_column=None):
        data_set = deepcopy(data)
        x = deepcopy(data_set[feature_columns]).values
        y = deepcopy(data_set[target_column]).values
        skf = StratifiedKFold(n_splits=self.out_cv, shuffle=True, random_state=self.seed)
        cv_index = np.zeros(len(y), dtype=int)
        y_prob = np.zeros(len(y))
        y_pep = np.zeros(len(y))
        feature_importance_df = pd.DataFrame()
        for index, (train_index, test_index) in enumerate(skf.split(x, y)):
            print('++++++++++++++++CV {}+++++++++++++++'.format(index + 1))
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lgbbo = LGB_bayesianCV(n_splits=self.inner_cv)
            lgbbo.fit(x_train, y_train)
            y_testprob = lgbbo.predict_proba(x_test)[:, 1]
            y_prob[test_index] = y_testprob
            cv_index[test_index] = index + 1
            print('train auc:', lgbbo.best_auc)  # best val auc 或 lgbbo.model.best_score
            print('test auc:', roc_auc_score(y_test, y_testprob))
            y_pep[test_index] = Prob2PEP(y_test, y_testprob)

            fold_importance_df = pd.DataFrame()
            fold_importance_df["Feature"] = feature_columns
            fold_importance_df["Importance"] = lgbbo.model.feature_importances_
            fold_importance_df["cv_index"] = index + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        y_qvalue = Score2Qval(y, -y_pep)
        self.feature_imp = feature_importance_df[['Feature', 'Importance']].groupby(
            'Feature').mean().reset_index().sort_values(by='Importance')
        data_set['cv_index'] = cv_index
        data_set['Lgb_score'] = y_prob
        data_set['Lgb_pep'] = y_pep
        data_set['psm_qvalue'] = y_qvalue
        if protein_column is not None:
            data_set = GroupProteinPEP2Qval(data_set, file_column=file_column,
                                            protein_column=protein_column,
                                            target_column=target_column,
                                            pep_column='Lgb_pep')
        self.data_set = data_set
        return data_set

    def Feature_imp_plot(self):
        plt.figure(figsize=(6, 6))
        plt.barh(self.feature_imp.Feature, self.feature_imp.Importance, height=0.7, orientation="horizontal")
        plt.ylim(0, self.feature_imp.Feature.shape[0] - 0.35)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('Importance', fontsize=15)
        plt.ylabel('Features', fontsize=15)
        plt.tight_layout()

    def CVROC(self):
        index_column = 'cv_index'
        target_column = 'label'
        pred_column = 'Lgb_score'
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 10000)
        plt.figure(figsize=(5, 4))
        for i in sorted(self.data_set[index_column].unique()):
            y_true = self.data_set.loc[self.data_set[index_column] == i, target_column]
            y_prob = self.data_set.loc[self.data_set[index_column] == i, pred_column]

            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            mean_tpr += sp.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='Fold{} AUC = {}'.format(i, round(roc_auc, 4)))

        mean_tpr = mean_tpr / len(self.data_set[index_column].unique())
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, label='Mean AUC = {}'.format(round(mean_auc, 4)))
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('True Positive Rate', fontsize=15)
        plt.legend(fontsize=12)
        plt.tight_layout()

    def PSM_accept(self):
        data = self.data_set[self.data_set['psm_qvalue'] <= 0.05]
        data = data.sort_values(by='psm_qvalue')
        data['Number of PSMs'] = range(1, data.shape[0] + 1)

        plt.figure(figsize=(5, 4))
        plt.plot(data['psm_qvalue'], data['Number of PSMs'], label='DeepSCP')
        plt.axvline(x=0.01, ls="--", c="gray")
        plt.xlim(-0.001, 0.05)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('PSM q-value', fontsize=15)
        plt.ylabel('Number of PSMs', fontsize=15)
        plt.legend(fontsize=13)
        plt.tight_layout()


def PSM2ProPep(data, file_column,
               protein_column,
               peptide_column,
               intensity_columns):
    df = data[[file_column] + [protein_column] + [peptide_column] + intensity_columns]

    proteins = df.groupby([file_column, protein_column])[intensity_columns].sum(1).reset_index()
    df_pro = []
    for i, j in proteins.groupby([file_column]):
        k = pd.DataFrame(data=j[intensity_columns].values, index=j[protein_column].tolist(),
                         columns=['{}_{}'.format(i, x) for x in intensity_columns])
        df_pro.append(k)
    df_pro = pd.concat(df_pro, axis=1)
    df_pro[df_pro.isna()] = 0
    df_pro.index.name = protein_column

    peptides = df.groupby([file_column, protein_column, peptide_column])[intensity_columns].sum(1).reset_index()
    df_pep = []
    for i, j in peptides.groupby([file_column]):
        k = j.drop(file_column, axis=1).set_index([protein_column, peptide_column])
        k.columns = ['{}_{}'.format(i, x) for x in intensity_columns]
        df_pep.append(k)
    df_pep = pd.concat(df_pep, axis=1)
    df_pep[df_pep.isna()] = 0
    df_pep = df_pep.reset_index()
    return df_pro, df_pep


def proteinfilter(data, protein_count=15, sample_ratio=0.5):
    nrow = (data != 0).sum(1)
    data = data.loc[nrow[nrow >= protein_count].index]
    ncol = (data != 0).sum(0)
    data = data[ncol[ncol >= ncol.mean() * sample_ratio].index]
    return data


def main(evidenve_file, msms_file, lbmsms_file):
    print(' ###################SampleRT###################')
    evidence = pd.read_csv(evidenve_file, sep='\t', low_memory=False)
    sampleRT = MQ_SampleRT()
    dfRT = sampleRT.fit_tranform(evidence)
    del evidence
    print('###################DeepSpec###################')
    msms = pd.read_csv(lbmsms_file, sep='\t', low_memory=False)
    lbmsms = pd.read_csv(msms_file, sep='\t', low_memory=False)
    deepspec = DeepSpec()
    deepspec.fit(lbmsms)
    dfSP = deepspec.predict(dfRT, msms)
    del msms, lbmsms
    print('###################LgbBayses###################')
    dfdb = deepcopy(dfSP)
    del dfSP
    feature_columns = ['Length', 'Acetyl (Protein N-term)', 'Oxidation (M)', 'Missed cleavages',
                       'Charge', 'm/z', 'Mass', 'Mass error [ppm]', 'Retention length', 'PEP',
                       'MS/MS scan number', 'Score', 'Delta score', 'PIF', 'Intensity',
                       'Retention time', 'RT(*|rev)', 'RT(*|tag)', 'DeltaRT', 'PEPRT', 'ScoreRT',
                       'Cosine', 'PEPCosine', 'ScoreCosine']
    target_column = 'label'
    file_column = 'Experiment'
    protein_column = 'Leading razor protein'
    lgs = LgbBayes()
    data_set = lgs.fit_tranform(data=dfdb,
                                feature_columns=feature_columns,
                                target_column=target_column,
                                file_column=file_column,
                                protein_column=protein_column)
    data_set.to_csv('DeepSCP_evidence.txt', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepSCP: utilizing deep learning to boost single-cell proteome coverage")
    parser.add_argument("-e",
                        "--evidence",
                        dest='e',
                        type=str,
                        help="SCP SampleSet, evidence.txt, which recorde information about the identified peptides \
                        by MaxQuant with setting  FDR to 1 at both PSM and protein levels")
    parser.add_argument("-m",
                        "--msms",
                        dest='m',
                        type=str,
                        help="SCP SampleSet, msms.txt, which recorde fragment ion information about the identified peptides \
                        by MaxQuant with setting  FDR to 1 at both PSM and protein levels")
    parser.add_argument("-lbm",
                        "--lbmsms",
                        dest='lbm',
                        type=str,
                        help="LibrarySet, msms.txt, which recorde fragment ion information about the identified peptides \
                        by MaxQuant with setting  FDR to 0.01 at both PSM and protein levels")
    args = parser.parse_args()
    evidenve_file = args.e
    msms_file = args.m
    lbmsms_file = args.lbm
    t0 = time()
    main(evidenve_file, msms_file, lbmsms_file)
    print('DeepSCP using time: {} m {}s'.format(int((time() - t0) // 60), (time() - t0) % 60))
