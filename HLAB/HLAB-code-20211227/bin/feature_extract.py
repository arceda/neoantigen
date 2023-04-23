import torch
from transformers import BertModel, BertTokenizer, BertConfig
from bin.model_utils import ProteinBertSequenceClsRnn
import re
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

fatherDir = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/'
fatherpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class Feature_Extract():
    def __init__(self, split="train", models_path_or_name="./models", max_length=51):
        self.split = split
        self.datasetFolderPath = os.path.join(fatherDir, 'example/')
        self.trainFilePath = os.path.join(self.datasetFolderPath, 'example_train.csv')
        self.testFilePath = os.path.join(self.datasetFolderPath, 'example_test.csv')
        self.validFilePath = os.path.join(self.datasetFolderPath, 'example_valid.csv')

        self.trainSavePath = os.path.join(self.datasetFolderPath, 'train_embed.h5')
        self.testSavePath = os.path.join(self.datasetFolderPath, 'test_embed.h5')
        self.validSavePath = os.path.join(self.datasetFolderPath, 'valid_embed.h5')

        self.tokenizer = BertTokenizer.from_pretrained(models_path_or_name, do_lower_case=False)
        self.config = BertConfig.from_pretrained(
            models_path_or_name,
            num_labels=2,
        )

        self.config.rnn = "lstm"
        self.config.num_rnn_layer = 2
        self.config.rnn_dropout = 0.1
        self.config.rnn_hidden = 768
        self.config.length = 51
        self.model = ProteinBertSequenceClsRnn.from_pretrained(models_path_or_name,config=self.config)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        if split == "train":
            self.seqs, self.labels = self.load_dataset(self.trainFilePath)
            self.savepath = self.trainSavePath
        elif split == "valid":
            self.seqs, self.labels = self.load_dataset(self.validFilePath)
            self.savepath = self.validSavePath
        elif split == "test":
            self.seqs, self.labels = self.load_dataset(self.testFilePath)
            self.savepath = self.testSavePath

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

    def read_and_prepare(self, file):
        data = pd.read_csv(file)
        seqs = self.HLA_trans()
        data['cost_cents'] = data.apply(
            lambda row: self.transform(
                HLA=seqs[row['HLA'][0:5]+row['HLA'][6:]],
                peptide=row['peptide']),
            axis=1)
        return np.vstack(data.cost_cents)

    def get_label(self, file):
        data = pd.read_csv(file)
        label = []
        label.append(data['Label'].values)
        return label

    def load_dataset(self, data_path):
        file = data_path
        df = pd.read_csv(file)
        y_label = self.get_label(file)[0]
        X_test = self.read_and_prepare(file)
        X_test = X_test.tolist()
        X_test = [' '.join(eachseq) for eachseq in X_test]
        X_test = [" ".join(eachseq) for eachseq in
                  X_test]
        return (X_test, y_label)

    # Use the best performing model in the feature building block to extract new features from the original amino acid sequence.
    # Save the features in .h5 file.
    def embedding(self):
        BertEmbed = []
        SeqName = []
        for sequences in tqdm(self.seqs, desc=self.split):
            sequences = [re.sub(r"[UZOBJ]", "X", sequences)]
            ids = self.tokenizer.batch_encode_plus(sequences, max_length=51, add_special_tokens=True, padding=True)
            input_ids = torch.tensor(ids['input_ids']).to(self.device)
            attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
            # Extracting sequences' features and load it into the CPU if needed
            with torch.no_grad():
                embedding = self.model(input_ids=input_ids, attention_mask=attention_mask)[1]  #rnn—embedding
                embedding = embedding.cpu().numpy()
                # Remove padding ([PAD]) and special tokens ([CLS],[SEP]) that is added by Bert model
                SeqName.append(sequences[0])
                BertEmbed.append(embedding.tolist())

        Bert_feature = pd.DataFrame(BertEmbed)
        col = ["Bert_F" + str(i + 1) for i in range(0, 1536)]
        Bert_feature.columns = col
        Bert_feature.index = SeqName
        Bert_feature["label"] = self.labels
        print(Bert_feature)
        Bert_feature.to_hdf(self.savepath, 'a', complevel=4, complib='blosc', append=True)

