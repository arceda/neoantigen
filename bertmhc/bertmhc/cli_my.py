# archivo principal, aquí se empieza el entrenamiento
import sys
from utils_model import EarlyStopping, MAData
from tape import ProteinBertConfig
from dataloader import BertDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from bertmhc import BERTMHC, BERTMHC_LINEAR, BERTMHC_RNN, BERTMHC_RNN2, BERTMHC_RNN_ATT
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils_model import train, evaluate

# dataset ###########################################################################
trainset = BertDataset('../../dataset/netMHCIIpan3.2/train_mini.csv', max_pep_len=24)
valset = BertDataset('../../dataset/netMHCIIpan3.2/eval_mini.csv', max_pep_len=24)

train_data = DataLoader(        trainset,
                                batch_size=32,
                                shuffle=True,
                                num_workers=16,
                                pin_memory=True,
                                collate_fn=trainset.collate_fn)

val_data = DataLoader(        valset,
                              batch_size=64,
                              num_workers=16,
                              pin_memory=True,
                              collate_fn=valset.collate_fn)

logging.basicConfig(format='%(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Training on {0} samples, eval on {1}".format(len(trainset), len(valset)))
# model ###########################################################################
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BERTMHC_RNN2.from_pretrained('bert-base')

for p in model.bert.parameters():
    p.requires_grad = True

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)


# train ###########################################################################
epochs = 20
lr = 0.15
w_pos = 1.0 # mass positive weight
save = "TRAIN_7_bertmhc_model.pt"
alpha = 0.0 # alpha weight on mass loss, affinity loss weight with 1-alpha
patience = 5 # Earlystopping patience
metric = 'val_auc' # validation metric, default auc

aff_criterion = nn.BCEWithLogitsLoss() # Sigmoid layer and the BCELoss in one single class
w_pos = torch.tensor([w_pos]).to(device)
mass_criterion = nn.BCEWithLogitsLoss(pos_weight=w_pos, reduction='none')

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2, min_lr=1e-4, factor=0.1)

early_stopping = EarlyStopping(patience=patience, verbose=True, saveto=save)

for epoch in range(epochs):
    print("Training epoch {}".format(epoch))
    train_metrics = train(model, optimizer, train_data, device, aff_criterion, mass_criterion, alpha, scheduler)
    eval_metrics = evaluate(model, val_data, device, aff_criterion, mass_criterion, alpha)
    eval_metrics['train_loss'] = train_metrics
    logs = eval_metrics

    scheduler.step(logs.get(metric))
    logging.info('Sample dict log: %s' % logs)

    # callbacks
    early_stopping(-logs.get(metric), model, optimizer)
    if early_stopping.early_stop or logs.get(metric) <= 0:
        print("Early stopping")
        break