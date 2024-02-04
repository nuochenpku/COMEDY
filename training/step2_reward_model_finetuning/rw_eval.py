#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import torch
from tqdm import tqdm
import deepspeed
import json

from transformers import AutoTokenizer
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.utils import to_device

# local_rank = int(os.getenv('LOCAL_RANK', '0'))
# world_size = int(os.getenv('WORLD_SIZE', '1'))

def load_stuff(model_name_or_path, num_padding_at_beginning=0):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=True)

    model = create_critic_model(model_name_or_path, tokenizer, None,
                                num_padding_at_beginning, True)

    return model, tokenizer


def prepare_singlbatch(prompt_list,
                         ans_list,
                         tokenizer,
                         max_seq_len=1024,
                         end_of_conversation_token="[EOS]"):
    chosen_sentence = [prompt + ans + end_of_conversation_token for prompt, ans in zip(prompt_list, ans_list)]
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = chosen_token["input_ids"]
    batch["attention_mask"] = chosen_token["attention_mask"]

    return batch


def infer_rewards(rm_model, datas, args):
    results = []
    for raw, batch in tqdm(datas):
        with torch.no_grad():
            # outputs = rm_model(
            #     **batch, prompt_length=max(2, args.num_padding_at_beginning)
            # ) 
            outputs = rm_model.forward_value(
                **batch, prompt_length=max(2, args.num_padding_at_beginning)
            ) 
            score = outputs["chosen_end_scores"].tolist()
        results.append((raw, score))
    return results


def save_to_file(results, output_file):
    with open(output_file, 'w', encoding='utf8') as fw:
        for jd, score in results:
            jd['rewards'] = list(map(float, score))
            js = json.dumps(jd, ensure_ascii=False)
            fw.write(f'{js}\n')


if __name__ == "__main__":

    # hyper parqams
    class args:
        # model_name_or_path = '/xllm3-ft/weiyihao/reward/2023-05-28-22.26.29/checkpoint-600'
        # model_name_or_path = '/xllm3-ft/weiyihao/reward/2023-05-29-10.49.17/checkpoint-1500'
        # model_name_or_path = '/xllm3-ft/junhuachen/outputs/dschat-reward/2023-05-29-13.25.53/checkpoint-512'
        # model_name_or_path = '/xllm3-ft/junhuachen/outputs/dschat-reward/2023-05-29-13.25.53/checkpoint-2048'
        # model_name_or_path = '/xllm3-ft/weiyihao/reward/2023-05-31-20.06.13/checkpoint-900' # v2-150k
        # model_name_or_path = '/xllm3-ft/weiyihao/reward/2023-05-31-19.51.44/checkpoint-2000' # v2-150k+fulleng
        # model_name_or_path = '/xllm3-ft/junhuachen/outputs/dschat-reward/2023-06-01-16.58.37/checkpoint-1666'
        model_name_or_path = '/xllm3-ft/junhuachen/outputs/dschat-reward/2023-06-01-16.58.37/checkpoint-3332'
        num_padding_at_beginning = 0
    # device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"]='6'
    device='cuda:0'

    # datafile
    # input_file = '/xllm3-ft/share/general-benchmark/reward/xiaoice_ranking_benchmark.jsonl'
    input_file = '/xllm3-ft/share/general-benchmark/reward/zbench_ranking_benchmark.jsonl'
    output_file = f'{input_file}.reward-ziyaD-v2-3332'
    print(output_file)

    # loading model
    rm_model, tokenizer = load_stuff(args.model_name_or_path,
                                     args.num_padding_at_beginning)
    rm_model.to(device)
    # rm_model = deepspeed.init_inference(
    #     rm_model,
    #     mp_size=world_size,
    #     dtype=torch.half,
    #     replace_method='auto',
    #     replace_with_kernel_inject=True
    # )

    # get datas
    datas = []
    with open(input_file, 'r', encoding='utf8') as f:
        for row in tqdm(f):
            jd = json.loads(row)
            human_pf = jd['human_prefix']
            ai_pf = jd['ai_prefix']
            bos = '[BOS]'
            eos = '[EOS]'
            ans_list = [item['text'] for item in jd['respsonses']]
            prompt = [
                f"{bos}{human_pf}{item['content']}{eos}" if item['role']=='Human' else f"{bos}{ai_pf}{item['content']}{eos}"
                for item in jd['session']]
            prompt = '\n'.join(prompt) + f'\n{bos}{ai_pf}' 
            jd['prompt'] = prompt # add prompt to results
            prompt_list = [prompt] * len(ans_list)
            batch = prepare_singlbatch(
                prompt_list,
                ans_list,
                tokenizer,
                max_seq_len=2048,
                end_of_conversation_token="[EOS]")
            batch = to_device(batch, device)
            datas.append((jd, batch))
    
    # infer
    results = infer_rewards(rm_model, datas, args)
    
    # save
    save_to_file(results, output_file)






