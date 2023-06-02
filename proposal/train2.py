import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch.nn import MSELoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.bert.modeling_bert import BERT_INPUTS_DOCSTRING, BERT_START_DOCSTRING
from transformers import Trainer, TrainingArguments, BertConfig

from tape.tokenizers import TAPETokenizer
from tape import ProteinBertConfig

from data_loader import My_Load_Dataset
from transformers import EarlyStoppingCallback, IntervalStrategy
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score


import math

def compute_metrics(pred):
    labels = pred.label_ids
    prediction=pred.predictions
    preds = prediction.argmax(-1)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    precision = tp / (tp + fp) 
    recall = tp / (tp + fn)
    sn = tp / (tp + fp)       
    sp = tn / (tn + fp)  # true negative rate
    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sn': sn,
        'sp': sp,
        'accuracy': acc,
        'mcc': mcc
    }

model_name = "bert-base"

train_dataset = My_Load_Dataset(path="../dataset/netMHCIIpan3.2/train_micro.csv", tokenizer_name=model_name, max_length=71)
val_dataset = My_Load_Dataset(path="../dataset/netMHCIIpan3.2/eval_micro.csv", tokenizer_name=model_name, max_length=71)

print(train_dataset[0]['input_ids'].shape)
print(val_dataset[0])
print(test_dataset[0])

config = ProteinBertConfig.from_pretrained("bert-base", num_labels=2)
#config = BertConfig.from_pretrained("bert-base", num_labels=2)
config.rnn = "lstm"
config.num_rnn_layer = 2
config.rnn_dropout = 0.1
config.rnn_hidden = 768
config.length = 51
config.cnn_filters = 512
config.cnn_dropout = 0.1
