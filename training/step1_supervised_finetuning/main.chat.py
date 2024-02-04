#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils import tensorboard

from datasets.distributed import split_dataset_by_node

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaTokenizer,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils_chat import create_prompt_dataset, get_unsupervised_data
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model

# ntk 
from utils.model.llama_condense_monkey_patch import replace_llama_with_condense
from utils.model.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='6,2,2',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `2,4,4`'
                        'will use 60% of data for phase 1, 20% for phase 2'
                        'and 20% for phase 3.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.1,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument('--tensorboard_path', type=str, default='./tensorboard_log/')
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--train_data_path', type=str, required=True, help="training path")
    parser.add_argument('--valid_data_path', type=str, required=True, help="validation path")
    parser.add_argument('--num_train_samples', type=int, required=True, help="number of rows in training datas.")
    parser.add_argument('--save_every_epoch', action='store_true')
    parser.add_argument('--ntk_RoPE_scaling_ratio', type=int, default=1, help="scaling context window.")
    parser = deepspeed.add_config_arguments(parser)
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    print_rank_0(args)
    print_rank_0(unknown)

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimizer_lora cannot be enabled at the same time."

    return args


def main():
    args = parse_args()

    if args.tensorboard_path != "":
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    assert not args.offload, "zero-offload is not currently supported but coming soon!"

    torch.distributed.barrier()

    # ntk rope scaling
    if args.ntk_RoPE_scaling_ratio > 1:
        replace_llama_with_condense(ratio=args.ntk_RoPE_scaling_ratio)
        replace_llama_attn_with_flash_attn()

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path,
                                              fast_tokenizer=True)
    # special_tokens_dict = {'additional_special_tokens': ["<s>","</s>","<unk>"]} # 其实这行可有可无
    # tokenizer.add_special_tokens(special_tokens_dict) 
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    # tokenizer.padding_side = 'left' # 其他的instft都默认用了right pad

    model = create_hf_model(AutoModelForCausalLM, args.model_name_or_path,
                            tokenizer, ds_config)

    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)

    # Prepare the data
    train_phase = 1
    # train_dataset, eval_dataset = create_prompt_dataset(
    #     args.local_rank, args.data_path, args.data_split,
    #     args.data_output_path, train_phase, args.seed, tokenizer,
    #     args.max_seq_len)
    train_dataset = get_unsupervised_data(args, tokenizer, args.train_data_path, train_phase=train_phase)
    eval_dataset = get_unsupervised_data(args, tokenizer, args.valid_data_path, train_phase=train_phase)

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        model.train()
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        try:
            loss = get_all_reduce_mean(losses).item()
        except:
            loss = float("inf")
        return perplexity, loss

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    perplexity, dev_loss = evaluation(model, eval_dataloader)
    print_rank_0(f"ppl: {perplexity}, loss: {dev_loss}", args.global_rank)
    global_step = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            model.backward(loss)
            model.step()
            mean_loss += loss.item()
            total_steps = global_step + step
            if total_steps > 0 and total_steps % args.log_interval == 0:
                print_rank_0(
                    f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}",
                    args.global_rank)
                if args.global_rank == 0:
                    writer.add_scalar("Train/loss", mean_loss/(step+1),  total_steps)

            if total_steps > 0 and total_steps % args.eval_interval == 0:
                
                print_rank_0('evaluating model ...', args.global_rank)
                print_rank_0(
                    f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
                    args.global_rank)
                perplexity, dev_loss = evaluation(model, eval_dataloader)
                print_rank_0(f"ppl: {perplexity}, loss: {dev_loss}", args.global_rank)
                if args.global_rank == 0:
                    writer.add_scalar("Dev/perplexity", perplexity,  total_steps)
                    writer.add_scalar("Dev/Loss", dev_loss, total_steps)

            if total_steps > 0 and total_steps % args.save_interval == 0 and args.output_dir is not None:
                print_rank_0(f'saving model ... to step {total_steps}', args.global_rank)
                if args.lora_dim > 0:
                    model = convert_lora_to_linear_layer(model)
                if args.zero_stage == 3:
                    # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
                    save_zero_three_model(model,
                                        tokenizer,
                                        args.global_rank,
                                        os.path.join(args.output_dir, f'checkpoint-{total_steps}'),
                                        zero_stage=args.zero_stage)
                else:
                    if args.global_rank == 0:
                        save_hf_format(model, tokenizer, args, sub_folder=f'checkpoint-{total_steps}')
        
        global_step = total_steps

        if args.save_every_epoch:
            print_rank_0(f'saving model ... to step {total_steps}', args.global_rank)
            if args.lora_dim > 0:
                model = convert_lora_to_linear_layer(model)
            if args.zero_stage == 3:
                # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
                save_zero_three_model(model,
                                    tokenizer,
                                    args.global_rank,
                                    os.path.join(args.output_dir, f'checkpoint-{total_steps}'),
                                    zero_stage=args.zero_stage)
            else:
                if args.global_rank == 0:
                    save_hf_format(model, tokenizer, args, sub_folder=f'checkpoint-{total_steps}')

        model.tput_timer.update_epoch_count()

    # saving at the end
    if args.output_dir is not None and (not args.save_every_epoch):
        print_rank_0('saving model ...', args.global_rank)
        if args.lora_dim > 0:
            model = convert_lora_to_linear_layer(model)
        if args.zero_stage == 3:
            # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
            save_zero_three_model(model,
                                tokenizer,
                                args.global_rank,
                                args.output_dir,
                                zero_stage=args.zero_stage)
        else:
            if args.global_rank == 0:
                save_hf_format(model, tokenizer, args)

if __name__ == "__main__":
    main()
