# archivo principal, aquí se empieza el entrenamiento
import sys
from utils_model import EarlyStopping, MAData
from tape import ProteinBertConfig
from dataloader import BertDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from bertmhc import BERTMHC
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils_model import train, evaluate

logging.basicConfig(format='%(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Console script for bertmhc."""
    parser = argparse.ArgumentParser(description='PyTorch BERTMHC model')
    subparsers = parser.add_subparsers()

    # train
    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train_bertmhc)
    train_parser.add_argument('--data', type=str, default='../tests/data/',
                        help='location of the data corpus')
    train_parser.add_argument('--eval', type=str, default='eval.csv',
                        help='evaluation set')
    train_parser.add_argument('--train', type=str, default='train.csv',
                        help='training set')
    train_parser.add_argument('--peplen', type=int, default=22,
                        help='peptide epitope length')
    train_parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    train_parser.add_argument('--alpha', type=float, default=0.0,
                        help='alpha weight on mass loss, affinity loss weight with 1-alpha')
    train_parser.add_argument('--epochs', type=int, default=30,
                        help='upper epoch limit')
    train_parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size')
    train_parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay')
    train_parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    train_parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    train_parser.add_argument('--w_pos', type=float, default=1.0,
                        help='mass positive weight')
    train_parser.add_argument('--metric', type=str, default='val_auc',
                        help='validation metric, default auc')
    train_parser.add_argument('--random_init', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='If True, Initialize the model random')
    train_parser.add_argument('--calibrate', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Calibrate probability')
    train_parser.add_argument('--deconvolution', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='If True, need to give Single allele (SA) and multi-allele (MA) data')
    train_parser.add_argument('--patience', type=int, default=5,
                        help='Earlystopping patience')
    train_parser.add_argument('--sa_epoch', type=int, default=15,
                        help="Number of epochs to train with single-allele data before deconvolution starts")
    train_parser.add_argument('--instance_weight', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='If True, use instance weights from the input data frame')
    train_parser.add_argument('--negative', type=str, default='max',
                        help="'max: maximum predicted, 'all: use all negatives")

    # predict
    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(func=predict)

    predict_parser.add_argument('--data', type=str, default='tests/data/eval.csv',
                              help='location of the data to predict')
    predict_parser.add_argument('--model', type=str, default='model.pt',
                              help='path to the trained model file')
    predict_parser.add_argument('--batch_size', type=int, default=512,
                              help='batch size')
    predict_parser.add_argument('--peplen', type=int, default=22,
                              help='peptide epitope length')
    predict_parser.add_argument('--task', type=str, choices=['binding', 'presentation'],
                                help='which prediction task, binding or presentation')
    predict_parser.add_argument('--output', type=str, default='output.csv',
                                help='path to the output file')
    args = parser.parse_args()
    args.func(args)
    logging.info("Arguments: %s", args)


def train_bertmhc(args):

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device", device)

    ###############################################################################
    # Load data
    ###############################################################################    
    if args.deconvolution:
        print("Loading data .............. deconvolution")
        trainMa = MAData(args.data + args.train,
                         sa_epochs=args.sa_epoch,
                         calibrate=args.calibrate,
                         negative=args.negative)
        valMa = MAData(args.data + args.eval,
                       sa_epochs=args.sa_epoch,
                       calibrate=args.calibrate,
                       negative=args.negative)
    else:
        print("Loading data ..............")
        trainset = BertDataset(args.data + args.train,
                               max_pep_len=args.peplen,
                               instance_weight=args.instance_weight)
        valset = BertDataset(args.data + args.eval,
                             max_pep_len=args.peplen,
                             instance_weight=args.instance_weight)
        train_data = DataLoader(trainset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=16,
                                pin_memory=True,
                                collate_fn=trainset.collate_fn)
        val_data = DataLoader(valset,
                              batch_size=args.batch_size*2,
                              num_workers=16,
                              pin_memory=True,
                              collate_fn=valset.collate_fn)
        logger.info("Training on {0} samples, eval on {1}".format(len(trainset), len(valset)))

    ################
    # Load model
    ################
    if args.random_init:
        config = ProteinBertConfig.from_pretrained('bert-base')
        model = BERTMHC(config)
    else:
        print("\nCargamos los pesos de TAPE\n\n")
        model = BERTMHC.from_pretrained('bert-base')

    for p in model.bert.parameters():
        p.requires_grad = True

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    #print(model)

    # loss
    aff_criterion = nn.BCEWithLogitsLoss()
    w_pos = torch.tensor([args.w_pos]).to(device)
    mass_criterion = nn.BCEWithLogitsLoss(pos_weight=w_pos, reduction='none')

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, nesterov=True)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2, min_lr=1e-4, factor=0.1)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, saveto=args.save)

    for epoch in range(args.epochs):
        if args.deconvolution:
            trainset = BertDataset(trainMa.generate_training(model, args.peplen, score='mass_pred',
                                                             batch_size=args.batch_size*2),
                                   max_pep_len=args.peplen,
                                   instance_weight=args.instance_weight)
            valset = BertDataset(valMa.generate_training(model, args.peplen, score='mass_pred',
                                                         batch_size=args.batch_size*2),
                                 max_pep_len=args.peplen,
                                 instance_weight=args.instance_weight)
            train_data = DataLoader(trainset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=16,
                                    pin_memory=True,
                                    collate_fn=trainset.collate_fn)
            val_data = DataLoader(valset,
                                  batch_size=args.batch_size,
                                  num_workers=16,
                                  pin_memory=True,
                                  collate_fn=valset.collate_fn)
            trainMa.close()
            valMa.close()
            if epoch == trainMa.sa_epochs:
                print('Reset early stopping')
                # reset early stopping and scheduler
                early_stopping.reset()
                scheduler._reset()

        print("Training epoch {}".format(epoch))
        train_metrics = train(model, optimizer, train_data, device, aff_criterion, mass_criterion, args.alpha, scheduler)
        eval_metrics = evaluate(model, val_data, device, aff_criterion, mass_criterion, args.alpha)
        eval_metrics['train_loss'] = train_metrics
        logs = eval_metrics

        scheduler.step(logs.get(args.metric))
        logging.info('Sample dict log: %s' % logs)

        # callbacks
        early_stopping(-logs.get(args.metric), model, optimizer)
        if early_stopping.early_stop or logs.get(args.metric) <= 0:
            if args.deconvolution and not trainMa.train_ma:
                # still training SA only model, now switch to training on MA immediately
                trainMa.train_ma = True
                valMa.train_ma = True
                print("Start training with multi-allele data.")
            else:
                print("Early stopping")
                break

def predict(args):
    inp = args.data
    config = ProteinBertConfig.from_pretrained('bert-base')
    model = BERTMHC(config)
    weights = torch.load(args.model)
    if list(weights.keys())[0].startswith('module.'):
        weights = {k[7:]: v for k, v in weights.items() if k.startswith('module.')}
    model.load_state_dict(weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    valset = BertDataset(inp,
                         max_pep_len=args.peplen,
                         train=False)
    val_data = DataLoader(valset,
                          batch_size=args.batch_size,
                          num_workers=16,
                          pin_memory=True,
                          collate_fn=valset.collate_fn)
    pred = []
    for batch in tqdm(val_data):
        batch = {name: tensor.to(device)
                 for name, tensor in batch.items()}
        logits, _ = model(**batch)
        pred.append(torch.sigmoid(logits).cpu().detach().numpy())
    dt = pd.read_csv(inp)
    pred = np.concatenate(pred)
    if args.task == 'binding':
        dt['bertmhc_pred'] = pred[:,0]
    else:
        dt['bertmhc_pred'] = pred[:,1]
    dt.to_csv(args.output, index=None)
    return 0

if __name__ == "__main__":
    sys.exit(main())
