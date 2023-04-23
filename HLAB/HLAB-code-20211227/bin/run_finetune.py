from transformers import Trainer, TrainingArguments, BertConfig
from bin.model_utils import BertForSequenceClassification
from bin.data_generate import Load_Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from bin.args_utils import get_args


def compute_metrics(pred):
    '''

    :param pred:
    :return:
    '''
    labels = pred.label_ids
    prediction=pred.predictions
    preds = prediction.argmax(-1)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    return {
        'auc': auc,
        'sn': sn,
        'sp': sp,
        'accuracy': acc,
        'mcc': mcc
    }


if __name__ == '__main__':

    args = get_args()

    model_name = args.model_name_or_path  # 'Rostlab/prot_bert_bfd'
    max_seq_length = args.max_seq_length  # 51
    train_dataset = Load_Dataset(split="train", tokenizer_name=model_name, max_length=max_seq_length)
    val_dataset = Load_Dataset(split="valid", tokenizer_name=model_name, max_length=max_seq_length)
    test_dataset = Load_Dataset(split="test", tokenizer_name=model_name, max_length=max_seq_length)

    num_labels = 2
    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
    )
    config.rnn = args.rnn
    config.num_rnn_layer = args.num_rnn_layer
    config.rnn_dropout = args.rnn_dropout
    config.rnn_hidden = args.rnn_hidden
    config.length = args.max_seq_length
    config.cnn_filters = args.cnn_filters
    config.cnn_dropout = args.cnn_dropout

    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        num_train_epochs=args.num_train_epochs,  # total number of training epochs
        per_device_train_batch_size=args.per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.per_device_eval_batch_size,  # batch size for evaluation
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        learning_rate=args.lr,  # The initial learning rate for optimizer.
        logging_dir=args.logging_dir,  # directory for storing logs './logs'
        logging_steps=args.logging_steps,  # How often to print logs
        save_steps=args.save_steps,
        do_train=args.do_train,  # Perform training
        do_eval=args.do_eval,  # Perform evaluation
        eval_steps=args.eval_steps,  # How often to eval
        evaluation_strategy=args.evaluation_strategy,  # evalaute per eval_steps
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # total number of steps before back propagation
        fp16=args.fp16,  # Use mixed precision
        fp16_opt_level=args.fp16_opt_level,  # mixed precision mode
        run_name=args.run_name,  # experiment name
        seed=args.seed,  # Seed for experiment reproducibility 3x3
    )

    model = Trainer(
        # model_init=model_init,                # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        model=BertForSequenceClassification.from_pretrained(model_name, config=config),  # ProBERT
        # model=ProteinBertSequenceClsRnn.from_pretrained(model_name, config=config),       # ProBERT+BiLSTM
        # model=ProteinBertSequenceClsRnnAtt.from_pretrained(model_name, config=config),    # ProBERT+BiLSTM+Attention
        # model=ProteinBertSequenceClsCnn.from_pretrained(model_name, config=config),       # ProBERT+CNN
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # evaluation metrics
    )
    # model.train(resume_from_checkpoint="./checkpoint-25200")  #continue from checkpoint
    model.train()
    model.save_model('models/')
    predictions, label_ids, metrics = model.predict(test_dataset)
    print(metrics)