#!/bin/bash

# DeepSpeed Team

CURRENT_TIME=$(TZ=UTC-8 date +"%Y-%m-%d-%H.%M.%S")

ZERO_STAGE="--zero_stage 2"

MODEL_PATH=$1
OUTPUT=$2
LOG_PATH=$3

export TOKENIZERS_PARALLELISM=False
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# 记得先shuffle好！！！train data
TRN_FN=$4
DEV_FN=$5

TOTAL_SIZE=`wc -l ${TRN_FN}`
echo "number of samples in trainset: ${TOTAL_SIZE}"

mkdir -p $OUTPUT/$CURRENT_TIME
deepspeed --include localhost:0,1,2,3,4,5,6,7 \
--master_port 12390 \
training/step1_supervised_finetuning/main.py \
   --model_name_or_path ${MODEL_PATH} \
   --train_data_path ${TRN_FN} \
   --valid_data_path ${DEV_FN} \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --data_output_path $OUTPUT/data \
   --max_seq_len 2048 \
   --learning_rate 1e-5  \
   --weight_decay 0.1 \
   --num_train_epochs 3 \
   --num_train_samples ${TOTAL_SIZE} \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 400 \
   --seed 42 \
   ${ZERO_STAGE} \
   --save_interval 2000 \
   --log_interval 100 \
   --eval_interval 1000 \
   --output_dir $OUTPUT/$CURRENT_TIME \
   --gradient_checkpointing \
   --tensorboard_path $LOG_PATH \
   &>$OUTPUT/train.log&

   

      
   

