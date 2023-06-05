from transformers import Trainer, TrainingArguments, BertConfig, AdamW
from model_utils_bert import BertLinear, BertRnn, BertRnnAtt
from model_utils_tape import TapeLinear
from transformers import EarlyStoppingCallback, IntervalStrategy
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from tape import ProteinBertConfig
from torch.utils.data import DataLoader
from transformers import get_scheduler

# data loaders
from dataloader_bert import DataSetLoaderBERT
from dataloader_tape import DataSetLoaderTAPE

import sys

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


path_train_csv = "../dataset/netMHCIIpan3.2/train_mini.csv"
path_val_csv = "../dataset/netMHCIIpan3.2/eval_mini.csv"

#################################################################################
#################################################################################
# Especificar si usaremos tape o bert
model_type = "tape"
#model_type = "bert" # EM1, ESM2, PortBert

# especificar donde se guadra los modlos y resultados
path_results    = "results/train_100/" 
path_model      = "models/train_100/"

# el modelo preentrenado
model_name = "bert-base"   # TAPE                   # train 1, 2, 3, 4
#model_name = "../models/esm2_t6_8M_UR50D"          # train 1, 2, 3, 4
#model_name = "../models/esm2_t33_650M_UR50D"       # 
model_name = "../models/esm2_t30_150M_UR50D"
#################################################################################
#################################################################################

if model_type == "tape":
    # read with TAPE tokenizer, la longitus del mhc es 34 => 34 + 37 + 2= 73    
    trainset = DataSetLoaderTAPE(path_train_csv, max_pep_len=37, max_length=73) # el paper usa max_peptide_lenght = 24
    valset = DataSetLoaderTAPE(path_val_csv, max_pep_len=37, max_length=73)
    config = ProteinBertConfig.from_pretrained(model_name, num_labels=2)
    
else:
    # read with ESM tokenizer    
    trainset = DataSetLoaderBERT(path=path_train_csv, tokenizer_name=model_name, max_length=73)
    valset = DataSetLoaderBERT(path=path_val_csv, tokenizer_name=model_name, max_length=73)
    config = BertConfig.from_pretrained(model_name, num_labels=2)

config.rnn = "lstm"
config.num_rnn_layer = 2
config.rnn_dropout = 0.1
config.rnn_hidden = 768
config.length = 51
config.cnn_filters = 512
config.cnn_dropout = 0.1

#################################################################################
#################################################################################
model_ = TapeLinear.from_pretrained(model_name, config=config)

#################################################################################
#################################################################################

#dataset = DataLoader(trainset)
#iterator = iter(dataset)
#print(next(iterator))
#print(next(iterator))
#print(trainset[0]['input_ids'].shape)

#sys.exit()

############ hyperparameters ####################################################

num_samples = len(trainset)
num_epochs = 10
batch_size = 32
num_training_steps = num_epochs * num_samples

training_args = TrainingArguments(
        output_dir                  = path_results, 
        num_train_epochs            = num_epochs,   
        per_device_train_batch_size = batch_size,   
        per_device_eval_batch_size  = batch_size,         
        logging_dir                 = path_results,        
        logging_strategy            = "epoch",
        # for early stopping
        eval_steps                  = num_samples/batch_size, # How often to eval        
        metric_for_best_model       = 'f1',
        load_best_model_at_end      = True,        
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch"
    )

# hiperparameters BERTMHC, uso SGD with momentum, ademas uso escheduler
lr = 0.15
weight_decay = 0.0001
momentum = 0.9

# hiperparameters HLAB, uso AdamW (en teoria es mejor de SGD with momentum)
#lr = 5e-5
#weight_decay = 0.01
betas = ((0.9, 0.999)) # defult

# optimizer Adam Weigh Decay https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
optimizer = AdamW(model_.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

# scheduller
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

trainer = Trainer(        
        args            = training_args,   
        model           = model_, 
        train_dataset   = trainset,  
        eval_dataset    = valset, 
        compute_metrics = compute_metrics,  
        optimizers = (optimizers, lr_scheduler),        
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=5)] 
    )

#trainer.train(resume_from_checkpoint = True)
trainer.train()
#trainer.save_model(path_model)
