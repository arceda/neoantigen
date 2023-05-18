import torch
from transformers import BertTokenizer
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import re

from transformers import Trainer, TrainingArguments, BertConfig
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score

class My_Load_Dataset(Dataset):
    def __init__(self, path, tokenizer_name='esm2_t6_8M_UR50D', max_length=51):                  
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

        self.seqs, self.labels = self.load_dataset(path)        
        self.max_length = max_length

    # usadao por HLAB, nosotros ya tenemos las seudosecuencias
    """def HLA_trans(self):
        HLA_seq = pd.read_csv('source/MHC_pseudo.dat', sep='\t')
        seqs = {}
        for i in range(len(HLA_seq)):
            seqs[HLA_seq.HLA[i]] = HLA_seq.sequence[i]
        return seqs
    """
    def transform(self, HLA, peptide):
        data = HLA + peptide
        data = data + 'X' * (69 - len(data))  # 71 max peptide-mhc length in dataset
        return data

    def read_and_prepare(self,file):
        data = pd.read_csv(file)
        """ # de HLAB original
        seqs = self.HLA_trans()
        data['cost_cents'] = data.apply(
            lambda row: self.transform(
                HLA=seqs[row['HLA'][0:5]+row['HLA'][6:]],
                peptide=row['peptide']),
            axis=1)
        return np.vstack(data.cost_cents)"""
        data['cost_cents'] = data.apply(
            lambda row: self.transform(HLA=row['mhc'], peptide=row['peptide']), axis=1)
        return np.vstack(data.cost_cents)

    def get_label(self,file):
        data = pd.read_csv(file)
        label = []
        #label.append(data['Label'].values)
        label.append(data['masslabel'].values) # netMHCpan3.2 database
        return label

    def load_dataset(self,data_path):
        file = data_path
        df = pd.read_csv(file)
        y_label = self.get_label(file)[0]
        X_test = self.read_and_prepare(file)
        X_test = X_test.tolist()
        X_test = [' '.join(eachseq) for eachseq in X_test]
        X_test = [" ".join(eachseq) for eachseq in
                  X_test]  # ['Y D S E Y R N I F T N T D E S N L Y L S Y N Y Y T W A V D A Y T W Y H M M V I F R L M',.....,'Y D S E Y R N I F T N T D E S N L Y L S Y N Y Y T W A V D A Y T W Y N F L I K F L L I']

        return (X_test, y_label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = " ".join("".join(self.seqs[idx].split()))
        

        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])

        return sample


