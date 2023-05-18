import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import re

fatherDir = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'
fatherpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# model_name = 'Rostlab/prot_bert_bfd'
class Load_Dataset(Dataset):

    def __init__(self, split="train", tokenizer_name='Rostlab/prot_bert_bfd', max_length=51):
        """
        Args:

        """
        self.datasetFolderPath = os.path.join(fatherDir, 'dataset/')
        self.trainFilePath = os.path.join(self.datasetFolderPath, 'train_data.csv')
        self.testFilePath = os.path.join(self.datasetFolderPath, 'test_data.csv')
        self.validFilePath = os.path.join(self.datasetFolderPath, 'valid_data.csv')

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

        if split == "train":
            self.seqs, self.labels = self.load_dataset(self.trainFilePath)
        elif split == "valid":
            self.seqs, self.labels = self.load_dataset(self.validFilePath)
        elif split == "test":
            self.seqs, self.labels = self.load_dataset(self.testFilePath)
        self.max_length = max_length

    def HLA_trans(self):
        HLA_seq = pd.read_csv('%s/source/MHC_pseudo.dat' % fatherpath, sep='\t')
        seqs = {}
        for i in range(len(HLA_seq)):
            seqs[HLA_seq.HLA[i]] = HLA_seq.sequence[i]
        return seqs

    def transform(self, HLA, peptide):
        data = HLA + peptide
        data = data + 'X' * (49 - len(data))
        return data

    def read_and_prepare(self,file):
        data = pd.read_csv(file)
        seqs = self.HLA_trans()
        data['cost_cents'] = data.apply(
            lambda row: self.transform(
                HLA=seqs[row['HLA'][0:5]+row['HLA'][6:]],
                peptide=row['peptide']),
            axis=1)
        return np.vstack(data.cost_cents)

    def get_label(self,file):
        data = pd.read_csv(file)
        label = []
        label.append(data['Label'].values)
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
        seq = re.sub(r"[UZOBJ]", "X", seq).upper()

        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])

        return sample



#################################################################
# Mi dataloader adaptado a la BD de netMHCpan3.2
#################################################################
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import re

class My_Load_Dataset(Dataset):
    def __init__(self, path, tokenizer_name='../../../models/prot_bert_bfd', max_length=51):                  
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

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
        data = data + 'X' * (49 - len(data)) # no usa el max length
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
        seq = re.sub(r"[UZOBJ]", "X", seq).upper()

        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])

        return sample
