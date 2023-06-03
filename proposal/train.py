from transformers import Trainer, TrainingArguments, BertConfig
from model_utils_bert import BertLinear, BertRnn, BertRnnAtt
from model_utils_tape import TapeLinear
from transformers import EarlyStoppingCallback, IntervalStrategy
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from tape import ProteinBertConfig
from torch.utils.data import DataLoader

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


path_train_csv = "../dataset/netMHCIIpan3.2/train_micro.csv"
path_val_csv = "../dataset/netMHCIIpan3.2/eval_micro.csv"

#################################################################################
#################################################################################
# Especificar si usaremos tape o bert
model_type = "tape"
#model_type = "bert" # EM1, ESM2, PortBert

# especificar donde se guadra los modlos y resultados
path_results    = "results/train_100/" # prueba de continuar el entrenamiento solo desde el ultimo checkpoint
path_model      = "models/train_100/"

# el modelo preentrenado
model_name = "bert-base"   # TAPE
#model_name = "../models/esm2_t6_8M_UR50D"
#model_name = "../models/esm2_t33_650M_UR50D"
#model_name = "../models/esm2_t30_150M_UR50D"

#################################################################################
#################################################################################

if model_type == "tape":
    # read with TAPE tokenizer, la longitus del mhc es 34 => 34 + 37 + 2= 73    
    trainset = DataSetLoaderTAPE(path_train_csv, max_pep_len=37, max_length=73) # el paper usa max_peptide_lenght = 24
    valset = DataSetLoaderTAPE(path_val_csv, max_pep_len=37, max_length=7)
    config = ProteinBertConfig.from_pretrained(model_name, num_labels=2)
    
else:
    # read with ESM tokenizer    
    trainset = DataSetLoaderBERT(path=path_train_csv, tokenizer_name=model_name, max_length=73)
    valset = DataSetLoaderBERT(path=path_val_csv, tokenizer_name=model_name, max_length=73)
    config = BertConfig.from_pretrained(model_name, num_labels=2)

#dataset = DataLoader(trainset)
#iterator = iter(dataset)
#print(next(iterator))
#print(next(iterator))


#print(trainset[0]['input_ids'].shape)

config.rnn = "lstm"
config.num_rnn_layer = 2
config.rnn_dropout = 0.1
config.rnn_hidden = 768
config.length = 51
config.cnn_filters = 512
config.cnn_dropout = 0.1
#print(config)

sys.exit()

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

num_samples = len(trainset)
num_epochs = 30
batch_size = 32

# con early stopping
training_args = TrainingArguments(
        output_dir                  = path_results, 
        num_train_epochs            = num_epochs,   
        per_device_train_batch_size = batch_size,   
        per_device_eval_batch_size  = batch_size,   
        warmup_steps                = 1000,         # number of warmup steps for learning rate scheduler
        weight_decay                = 0.01,         # strength of weight decay = L2 regulrization
        learning_rate               = 5e-5,         # The initial learning rate for optimizer.
        logging_dir                 = path_results, 
        gradient_accumulation_steps = 16,           # total number of steps before back propagation          
        logging_strategy            ="epoch",

        # a parir de aqui es necesario para early stopping
        eval_steps                  = num_samples/batch_size,          # How often to eval        
        metric_for_best_model       = 'f1',
        load_best_model_at_end      = True,        
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch"
    )

trainer = Trainer(        
        args            = training_args,   
        model           = BertLinear.from_pretrained(model_name, config=config), 
        train_dataset   = trainset,  
        eval_dataset    = valset, 
        compute_metrics = compute_metrics,  
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=5)] 
    )


#trainer.train(resume_from_checkpoint = True)
trainer.train()
#trainer.save_model(path_model)
