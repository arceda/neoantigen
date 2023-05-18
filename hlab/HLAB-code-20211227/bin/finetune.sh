export MODEL_NAME='Rostlab/prot_bert_bfd'
#export MODEL_NAME=./checkpoint-25200
export OUTPUT_DIR=./results_lr5e-5_bert_gas16_bs16
export LOGGING_DIR=./logging_lr5e-5_bert_gas16_bs16
export LR=5e-5
python run_finetune.py \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR\
    --logging_dir $LOGGING_DIR \
    --max_seq_length 51 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size=16   \
    --per_device_eval_batch_size=32   \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 16 \
    --lr $LR\
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --eval_steps 1052 \
    --save_steps 1052 \
    --logging_steps 1052 \
    --fp16