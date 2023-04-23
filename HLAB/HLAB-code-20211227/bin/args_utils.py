import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the log will be written.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model",
    )

    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=51,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to do prediction on the given dataset.")
    parser.add_argument(
        "--per_device_train_batch_size", default=64, type=int, help="Batch size  GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size", default=32, type=int, help="Batch size  GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=64,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Adam in bert layer.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument("--warmup_steps", default=1000, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="evalaute per eval_steps")
    parser.add_argument("--logging_steps", type=int, default=20, help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=50, help="Eval every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="save every X updates steps.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    # parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 for Adam optimizer.")
    # parser.add_argument("--beta2", default=0.999, type=float, help="Beta2 for Adam optimizer.")
    parser.add_argument("--rnn_dropout", default=0.1, type=float, help="Dropout rate of intermidiete layer.")
    parser.add_argument("--cnn_dropout", default=0.1, type=float, help="Dropout rate of intermidiete layer.")
    parser.add_argument("--rnn", default="lstm", type=str, help="What kind of RNN to use")
    parser.add_argument("--num_rnn_layer", default=2, type=int, help="Number of rnn layers in dnalong model.")
    parser.add_argument("--cnn_filters", default=512, type=int, help="Number of cnn filters in  model.")
    parser.add_argument("--rnn_hidden", default=768, type=int, help="Number of hidden unit in a rnn layer.")
    parser.add_argument("--seed", type=int, default=3, help="random seed for initialization")
    parser.add_argument("--fp16_opt_level", type=str, default="02", help="Apex AMP optimization level")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--run_name", type=str, default="Pro_Pep_bind", help="run name")
    args = parser.parse_args()
    return args