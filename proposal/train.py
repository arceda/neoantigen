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
from model_utils import ProteinBertSequenceClsRnnAtt, ProteinBertSequenceClsRnn, BertForSequenceClassification
from model_utils_tape import TapeLinear

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

#model_name = "../models/esm2_t33_650M_UR50D"
model_name = "../models/esm2_t30_150M_UR50D"

#train_dataset = My_Load_Dataset(path="../dataset/netMHCIIpan3.2/train_mini.csv", tokenizer_name=model_name, max_length=71)
#val_dataset = My_Load_Dataset(path="../dataset/netMHCIIpan3.2/eval_mini.csv", tokenizer_name=model_name, max_length=71)
#test_dataset = My_Load_Dataset(path="../dataset/netMHCIIpan3.2/test_mini.csv", tokenizer_name=model_name, max_length=71)

train_dataset = My_Load_Dataset(path="../dataset/netMHCIIpan3.2/train_micro.csv", tokenizer_name=model_name, max_length=71)
val_dataset = My_Load_Dataset(path="../dataset/netMHCIIpan3.2/eval_micro.csv", tokenizer_name=model_name, max_length=71)

print(train_dataset[0]['input_ids'].shape)
print(val_dataset[0])
print(test_dataset[0])

config = BertConfig.from_pretrained(model_name, num_labels=2)
#config = BertConfig.from_pretrained("bert-base", num_labels=2)
config.rnn = "lstm"
config.num_rnn_layer = 2
config.rnn_dropout = 0.1
config.rnn_hidden = 768
config.length = 51
config.cnn_filters = 512
config.cnn_dropout = 0.1


# model t6_8M 
# train_0 -> RNN_att con 3 apochs -> complete
# train_1 -> LINEAR con 20 apochs -> complete    -> early stopping
# train_2 -> RNN con 20 apochs -> complete
# train_3 -> RNN_att con 20 apochs -> complete   -> early stopping

# model t12_35M 
# train_4 -> LINEAR con 20 apochs -> complete    -> early stopping
# train_5 -> RNN con 20 apochs -> complete
# train_6 -> RNN_att con 20 apochs -> complete

# model t30_150M
# train_7 -> Linear con 20 epochs -> complete
# train_8 -> RNN con 20 epochs -> complete
# train_9 -> RNN_att con 20 epochs -> complete

# model t33_650M 
# train_10 -> Linear con 20 epochs -> 
# train_11 -> RNN con 20 epochs -> 
# train_12 -> RNN_att con 20 epochs -> 

#path_results = "results/train_9_test/" # prueba de continuar el entrenamiento solo desde el ultimo checkpoint
#path_model = "models/train_9_test/"


num_samples = 107424
num_epochs = 30
batch_size = 32

# con early stopping
training_args = TrainingArguments(
        output_dir                  = path_results, # output directory
        num_train_epochs            = num_epochs,            # total number of training epochs
        per_device_train_batch_size = batch_size,           # batch size per device during training
        per_device_eval_batch_size  = batch_size,           # batch size for evaluation
        warmup_steps                = 1000,         # number of warmup steps for learning rate scheduler
        weight_decay                = 0.01,         # strength of weight decay = L2 regulrization
        learning_rate               = 5e-5,         # The initial learning rate for optimizer.
        logging_dir                 = path_results, # directory for storing logs './logs'
        #logging_steps               = num_samples/batch_size,          # How often to print logs, cada epoch
        gradient_accumulation_steps = 16,           # total number of steps before back propagation  
        #save_steps                  = num_samples/batch_size,   
        logging_strategy="epoch",

        # a parir de aqui es necesario para early stopping
        eval_steps                  = num_samples/batch_size,          # How often to eval        
        metric_for_best_model       = 'f1',
        load_best_model_at_end      = True,
        #evaluation_strategy         = IntervalStrategy.STEPS, # "steps" # segun overleaf, pero cambiaremos 
        evaluation_strategy = "epoch",
        save_strategy = "epoch"
    )

model = Trainer(        
        args            = training_args,  # training arguments, defined above      
        model           = #.from_pretrained('bert-base', config=config),  # TAPE 
        #model           = BertForSequenceClassification.from_pretrained(model_name, num_labels=2),  # 
        #model           = ProteinBertSequenceClsRnn.from_pretrained(model_name, config=config),
        #model           = ProteinBertSequenceClsRnnAtt.from_pretrained(model_name, config=config),    # ProBERT+BiLSTM+Attention
        train_dataset   = train_dataset,  # training dataset
        eval_dataset    = val_dataset,  # evaluation dataset
        compute_metrics = compute_metrics,  # evaluation metrics
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=5)] # early stoping
    )


model.train(resume_from_checkpoint = True)
#model.train()
model.save_model(path_model)
