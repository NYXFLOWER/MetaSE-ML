#%%
from src.utils import *

import pandas as pd
# import numpy as np
from scipy import sparse as sp
# import torch
import pickle

# import plotly.express as px
# import plotly.graph_objects as go

from collections import OrderedDict
# import os

# from sklearn import metrics
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import LinearSVC


# torch.manual_seed(seed)		# fix seed for random numbers
# np.random.seed(100)

#%%
class MetaSEDataset(object):
    label_address = '../data/processed/drug-se-target.csv'
    dataset_dir = '../data/processed'

    def __init__(self, GEM_name):
        self.GEMout_address = f"../data/{GEM_name}/MLGEMs.txt"
        self.cid2drugid, self.reaction_fi, self.reaction_fa, self.fi, self.fa = self.__load_feature__()
        self.cid2seid, self.data_by_labels = self.__load_data__()
        self.n_drug = len(self.cid2drugid)

        with open(f'{self.dataset_dir}/secid2name.pkl', 'rb') as f:
            self.secid2name = pickle.load(f)

    def __load_data__(self):
        cid2seid = OrderedDict()
        data_se_drug = OrderedDict()

        da = pd.read_csv(self.label_address)
        for cid, ses, _ in da.values:
            if self.cid2drugid.get(cid) is None: 
                continue
            for se in ses.split(';'):
                if se == 'nan':
                    continue
                if cid2seid.get(se) is None:
                    cid2seid[se] = len(cid2seid)
                se = cid2seid[se]
                if data_se_drug.get(se) is None:
                    data_se_drug[se] = []
                
                data_se_drug[se].append(self.cid2drugid[cid])
        
        return cid2seid, data_se_drug

    def __repr__(self):
        return f"keys: {[i for i in self.__dict__.keys()]}"

    def __load_feature__(self):
        # load labels
        data = pd.read_csv(self.label_address, usecols=[0, 1])

        # load features
        gems = pd.read_csv(self.GEMout_address, header=None)
        cid2drugid, reaction2rid = OrderedDict(), OrderedDict()

        row, col, data_min, data_max = [], [], [], []
        for i in gems[0].values:
            [cid, fi, fa, r] = i.split()

            if cid2drugid.get(cid) is None:
                cid2drugid[cid] = len(cid2drugid)
            if reaction2rid.get(r) is None:
                reaction2rid[r] = len(reaction2rid)

            row.append(cid2drugid[cid])
            col.append(reaction2rid[r])
            data_min.append(float(fi))
            data_max.append(float(fa))
        
        n_drug, n_reaction = len(cid2drugid), len(reaction2rid)
        print(f'unique drug: {n_drug}')
        print(f'unique reaction: {n_reaction}')

        # generate sparse matrix for drug v.s. flux_min and flux_max
        fi_sp = sp.coo_matrix((data_min, (row, col)), shape=(n_drug, n_reaction))
        fa_sp = sp.coo_matrix((data_max, (row, col)), shape=(n_drug, n_reaction))
        
        fi, fa = fi_sp.toarray(), fa_sp.toarray()
        fi_st, fa_st = fi.std(axis=0), fa.std(axis=0)
        fi_idx, fa_idx = np.where(fi_st > 1e-12)[0], np.where(fa_st > 1e-12)[0]
        f_idx = np.unique(np.concatenate([fi_idx, fa_idx]))
        
        print(f'Effective reactions: \n  Flux_min: {fi_idx.shape[0]}\n  Flux_max: {fa_idx.shape[0]}')
        print(f'Unique effective reactions: {f_idx.shape[0]}')
        reactions = np.array(list(reaction2rid.keys()))
        
        return cid2drugid, reactions[fi_idx], reactions[fa_idx], fi[:, fi_idx], fa[:, fa_idx]

