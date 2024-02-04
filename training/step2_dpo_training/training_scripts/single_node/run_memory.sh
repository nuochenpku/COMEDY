
#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# local/xjsonfile/rftV2
# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2

if [ "$OUTPUT" == "" ]; then
    OUTPUT=/cpfs/user/chennuo/CN/output/compress_memory/13b_v2_dpo_0.01_sft/
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=29592 main.py  \
   --data_path /cpfs/user/chennuo/dsChatLLama/test_data/results/pair_200_sample_dpo_source.json \
   --data_split 0,10,0 \
   --model_name_or_path /cpfs/user/chennuo/CN/output/compress_memory/13b_v2/2023-12-28-00.12.51 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 2048 \
   --learning_rate 1e-5  \
   --weight_decay 0. \
   --num_train_epochs 2  \
   --beta 0.01 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 10 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --add_sft \
   --print_loss \
   --gradient_checkpointing \
   --output_dir $OUTPUT \
   --tensorboard_path $OUTPUT/runs \
   &> $OUTPUT/training.log 


#    9.65e-6
#    --gradient_checkpointing \