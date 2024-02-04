# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
import numpy as np
import os
from itertools import chain
from . import raw_datasets
from tqdm import tqdm
import json

DATASET_CACHE_DIR = "/xllm3-ft/data/cache/"

def get_raw_dataset(dataset_name, output_path, seed, local_rank, train_data_path=""):
    if dataset_name == "Dahoas/rm-static":
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank)
    elif dataset_name == "Dahoas/full-hh-rlhf":
        return raw_datasets.DahoasFullhhrlhfDataset(output_path, seed,
                                                    local_rank)
    elif dataset_name == "Dahoas/synthetic-instruct-gptj-pairwise":
        return raw_datasets.DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank)
    elif dataset_name == "yitingxie/rlhf-reward-datasets":
        return raw_datasets.YitingxieRlhfrewarddatasetsDataset(
            output_path, seed, local_rank)
    elif dataset_name == "openai/webgpt_comparisons":
        return raw_datasets.OpenaiWebgptcomparisonsDataset(
            output_path, seed, local_rank)
    elif dataset_name == "stanfordnlp/SHP":
        return raw_datasets.StanfordnlpSHPDataset(output_path, seed,
                                                  local_rank)
    elif dataset_name == "wangrui6/Zhihu-KOL":
        return raw_datasets.Wangrui6ZhihuKOLDataset(output_path, seed,
                                                    local_rank)
    elif dataset_name == "Cohere/miracl-zh-queries-22-12":
        return raw_datasets.CohereMiraclzhqueries2212Dataset(
            output_path, seed, local_rank)
    elif dataset_name == "Hello-SimpleAI/HC3-Chinese":
        return raw_datasets.HelloSimpleAIHC3ChineseDataset(
            output_path, seed, local_rank)
    elif dataset_name == "mkqa-Chinese":
        return raw_datasets.MkqaChineseDataset(output_path, seed, local_rank)
    elif dataset_name == "mkqa-Japanese":
        return raw_datasets.MkqaJapaneseDataset(output_path, seed, local_rank)
    elif dataset_name == "Cohere/miracl-ja-queries-22-12":
        return raw_datasets.CohereMiracljaqueries2212Dataset(
            output_path, seed, local_rank)
    elif dataset_name == "lmqg/qg_jaquad":
        return raw_datasets.LmqgQgjaquadDataset(output_path, seed, local_rank)
    elif dataset_name == "lmqg/qag_jaquad":
        return raw_datasets.LmqgQagjaquadDataset(output_path, seed, local_rank)
    elif dataset_name == 'xiaoice_label_datasets':
        return raw_datasets.XiaoiceLabelDataset(output_path, seed, local_rank, train_data_path)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(local_rank, output_path, dataset_name, seed,
                                split_name, data_split, split_index,
                                data_size):
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    if not os.path.isfile(index_file_name) and local_rank <= 0:
        splits = [float(s) for s in data_split.split(',')]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    torch.distributed.barrier()
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["input_ids"]
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id


def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    if train_phase == 1:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            if chosen_sentence is not None:
                chosen_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(
                    0)
                chosen_token["attention_mask"] = chosen_token[
                    "attention_mask"].squeeze(0)
                chosen_dataset.append(chosen_token)

    elif train_phase == 2:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            reject_sentence = raw_dataset.get_prompt_and_rejected(
                tmp_data)  # the accept response
            if chosen_sentence is not None and reject_sentence is not None:
                chosen_sentence += end_of_conversation_token  # the accept response
                reject_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                reject_token = tokenizer(reject_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"]
                chosen_token["attention_mask"] = chosen_token["attention_mask"]
                chosen_dataset.append(chosen_token)

                reject_token["input_ids"] = reject_token["input_ids"]
                reject_token["attention_mask"] = reject_token["attention_mask"]
                reject_dataset.append(reject_token)

    elif train_phase == 3:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            prompt = raw_dataset.get_prompt(tmp_data)
            if prompt is not None:
                output = tokenizer(prompt, add_special_tokens=False)
                prompt_token = {}
                prompt_token["input_ids"] = output["input_ids"]
                prompt_token["attention_mask"] = output["attention_mask"]
                nl_index = []
                for i in range(len(prompt_token["input_ids"])):
                    if prompt_token["input_ids"][i] == 7:
                        nl_index.append(i)
                if len(prompt_token["input_ids"]) > max_seq_len:
                    gap = len(prompt_token["input_ids"]) - max_seq_len
                    for i in nl_index:
                        if i - nl_index[0] > gap:
                            del prompt_token["input_ids"][nl_index[0]:i]
                            del prompt_token["attention_mask"][nl_index[0]:i]
                            break
                
                #PAD 到 max_seq_len
                add_count = max_seq_len-len(prompt_token["input_ids"])
                prompt_token["input_ids"] = [0] * add_count + prompt_token["input_ids"]
                prompt_token["attention_mask"] = [0] * add_count + prompt_token["attention_mask"]
                prompt_token = {
                    "input_ids": torch.tensor(prompt_token["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(prompt_token["attention_mask"], dtype=torch.long)
                }
                prompt_dataset.append(prompt_token)
    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


def create_dataset(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len, train_data_path=""):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank, train_data_path)
    print(len(raw_dataset))
    raise Exception('debug!!!')
    train_dataset = raw_dataset.get_train_data()
    eval_dataset = raw_dataset.get_eval_data()
    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                            raw_dataset.dataset_name_clean,
                                            seed, "train", data_split,
                                            train_phase - 1,
                                            len(train_dataset))
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                        train_phase, tokenizer,
                                        end_of_conversation_token,
                                        max_seq_len)

    
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                            raw_dataset.dataset_name_clean,
                                            seed, "eval",
                                            data_split, train_phase - 1,
                                            len(eval_dataset))
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)
    return train_dataset, eval_dataset


def create_prompt_dataset(local_rank,
                          data_path,
                          data_split,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="<|endoftext|>",
                          train_data_path=""):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = '_'.join(data_path)
    tokenizer_name = tokenizer.init_kwargs['name_or_path'].replace('/', '_')
    fname = '_'.join(fname.split('/'))
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    torch.distributed.all_reduce(buf_create_cache)

    # Skip creating cache if we found it on all the nodes.
    if buf_create_cache.item() == 0:
        return torch.load(train_fname), torch.load(eval_fname)
    else:
        train_dataset, eval_dataset = create_dataset(
            local_rank, data_path[0], data_split, output_path, train_phase,
            seed, tokenizer, end_of_conversation_token, max_seq_len, train_data_path=train_data_path)
        
        if local_rank <= 0:
            torch.save(train_dataset, train_fname)
            torch.save(eval_dataset, eval_fname)
        return train_dataset, eval_dataset


class DataCollatorReward:

    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0]
                                        for f in data] + [f[2] for f in data],
                                       dim=0)
        batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                            [f[3] for f in data],
                                            dim=0)
        return batch


class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        batch = {}
        batch["prompt"] = torch.stack([f[0] for f in data])
        batch["prompt_att_mask"] = torch.stack([f[1] for f in data])
        return batch


def infer_dataset_columns(datapath):
    with open(datapath, "r", encoding="utf-8") as file:
        line = file.readline()
        return list(json.loads(line).keys())


def get_prompt_data(args, tokenizer):

    data_files = args.train_data_path
    data_columns = infer_dataset_columns(data_files)

    prompt_raw_datasets = load_dataset("json", 
                                         data_files=data_files, 
                                         split="train", 
                                         name="prompt_dataset")
    
    max_len = args.max_prompt_seq_len

    def tokenize_function(examples):
        ret = {}
        text = examples['prompt'].replace('[END]','').replace('[BOS]','<s>').replace('[EOS]','</s>')
        output = tokenizer(text, add_special_tokens=False, 
                           max_length=max_len, truncation=True) # no padding here 
        ret["input_ids"] = output["input_ids"]
        ret["attention_mask"] = output["attention_mask"]

        if tokenizer.eos_token_id in ret["input_ids"]:
            ret["valid"] = 1
        else:
            ret["valid"] = 0

        # PAD 到 max_len
        pad_id = tokenizer.pad_token_id
        add_count = max_len-len(ret["input_ids"])
        ret["input_ids"] = [pad_id] * add_count + ret["input_ids"]
        ret["attention_mask"] = [0] * add_count + ret["attention_mask"]

        return ret

    # do tokenize
    prompt_raw_datasets = prompt_raw_datasets.map(tokenize_function, remove_columns=data_columns, num_proc=8)
    prompt_raw_datasets = prompt_raw_datasets.filter(lambda x: True if x["valid"] == 1 else False, num_proc=8)
    prompt_raw_datasets = prompt_raw_datasets.remove_columns(["valid"])
    # prompt_raw_datasets = prompt_raw_datasets.shuffle(seed=42)
    return prompt_raw_datasets


def get_reward_data(args, tokenizer, data_files):
    raw_datasets = load_dataset("json", 
                                data_files=data_files, 
                                split="train", 
                                name="reward_dataset")
    
    max_len = args.max_seq_len
    eos_token = tokenizer.eos_token

    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []

    def tokenize_function(examples):
        
        if '</s>' not in examples['ans']:
            examples['ans'] += '</s>'
        if '</s>' not in examples['rejected']:
            examples['rejected'] += '</s>'
        try:
            chosen_sentence = examples['prompt'] + examples['ans']
        except:
            chosen_sentence = examples['prompt'] + examples['chosen']
        reject_sentence = examples['prompt'] + examples['rejected']
        chosen_sentence = chosen_sentence.replace('[END]','').replace('[BOS]','<s>').replace('[EOS]','</s>')
        reject_sentence = reject_sentence.replace('[END]','').replace('[BOS]','<s>').replace('[EOS]','</s>')
        chosen_sentence = chosen_sentence if chosen_sentence.endswith(eos_token) else chosen_sentence + eos_token
        reject_sentence = reject_sentence if reject_sentence.endswith(eos_token) else reject_sentence + eos_token

        ret = {}
        chosen_token = tokenizer(chosen_sentence, add_special_tokens=False, padding="max_length",
                            max_length=max_len, truncation=True, return_tensors="pt")
        reject_token = tokenizer(reject_sentence, add_special_tokens=False, padding="max_length",
                            max_length=max_len, truncation=True, return_tensors="pt")

        if (chosen_token["input_ids"].view(-1).numpy()[-1] in {tokenizer.eos_token_id, tokenizer.pad_token_id}) and \
            (reject_token["input_ids"].view(-1).numpy()[-1] in {tokenizer.eos_token_id, tokenizer.pad_token_id}):
            ret['valid'] = 1
        else:
            ret['valid'] = 0

        ret["chosen_input_ids"] = chosen_token["input_ids"]
        ret["chosen_attention_mask"] = chosen_token["attention_mask"]
        ret["reject_input_ids"] = reject_token["input_ids"]
        ret["reject_attention_mask"] = reject_token["attention_mask"]
        
        return ret

    # do tokenize
    raw_datasets = raw_datasets.map(tokenize_function, remove_columns=['prompt','ans','rejected'], num_proc=10)
    raw_datasets = raw_datasets.filter(lambda x: True if x["valid"] == 1 else False, num_proc=10)
    raw_datasets = raw_datasets.remove_columns(["valid"])
    raw_datasets = raw_datasets.shuffle(seed=42)
    raw_datasets = raw_datasets.with_format("torch")

    for i, tmp_data in tqdm(enumerate(raw_datasets)):
        chosen_token = {
            "input_ids": tmp_data["chosen_input_ids"],
            "attention_mask": tmp_data["chosen_attention_mask"]
        }
        reject_token = {
            "input_ids": tmp_data["reject_input_ids"],
            "attention_mask": tmp_data["reject_attention_mask"]
        }
        chosen_dataset.append(chosen_token)
        reject_dataset.append(reject_token)

    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, 2)


def get_unsupervised_data(args, tokenizer, data_files, train_phase=1, streaming=False):
    data_columns = infer_dataset_columns(data_files)
    unsupervised_raw_datasets = load_dataset("json", 
                                            data_files=data_files, 
                                            streaming=streaming,
                                            split="train", 
                                            name="unsupdata")
    
    if train_phase==1:
        max_len = args.max_seq_len
    else:
        max_len = args.max_prompt_seq_len + args.max_answer_seq_len

    def _single_label_masking_phase1(label, bos_id):
        """
        find all bos position
        """
        bos_indices = [i for i,bi in enumerate(label) if bi==bos_id]
        if len(bos_indices) > 1:
            for i,(_start,_end) in enumerate(zip(bos_indices[:-1],bos_indices[1:])):
                if i%2==0:
                    for j in range(_start, _end):
                        label[j] = -100
        return label
    
    def _single_label_masking_phase3(label, bos_id):
        """
        find the last bos position
        """
        bos_indices = [i for i,bi in enumerate(label) if bi==bos_id]
        if len(bos_indices) > 1:
            _end = bos_indices[-1]
            for j in range(0, _end):
                label[j] = -100
        return label

    def _single_label_masking_alpaca(inputs, bos_id, eos_id, pad_id, maxlen):
        # return new input_ids, attention_mask and labels
        # 只保留AI的output
        labels = inputs.copy()
        bos_indices = [i for i,bi in enumerate(labels) if bi==bos_id]
        eos_indices = [i for i,bi in enumerate(labels) if bi==eos_id]

        # 只有一句话的doc，无需masking
        if len(bos_indices) == 1:
            input_ids = inputs # bos-eos必须成对出现
            new_labels = [-100] + labels[1:-1] + [-100]
            attention_mask = [1] * len(input_ids)
            # pad 0
            input_ids = (input_ids + [pad_id] * maxlen)[:maxlen]
            attention_mask = (attention_mask + [pad_id] * maxlen)[:maxlen]
            new_labels = (new_labels + [pad_id] * maxlen)[:maxlen]
            return input_ids, attention_mask, new_labels
        
        unmask_indices = [0] * len(inputs)
        if len(bos_indices) > 1:
            for i,(_start,_end) in enumerate(zip(bos_indices[:],eos_indices[:])):
                if i%2==1:
                    for j in range(_start+1, _end):
                        unmask_indices[j] = 1
        for j,umi in enumerate(unmask_indices):
            if umi==0:
                labels[j] = -100

        # 去掉中间的bos和eos，保留AI的eos
        input_ids = []
        new_labels = []
        flag = 0
        for x,y in zip(inputs, labels):
            if x == 1:
                continue
            elif x == 2:
                if flag % 2 == 0:
                    flag += 1
                    continue
                else:
                    flag += 1
                    input_ids.append(x)
                    new_labels.append(y)
            else:
                input_ids.append(x)
                new_labels.append(y)

        # 首尾加bos和eos，尾巴原数据已经有eos
        input_ids = [bos_id] + input_ids 
        new_labels = [-100] + new_labels 
        attention_mask = [1] * len(input_ids)

        # pad 0
        input_ids = (input_ids + [pad_id] * maxlen)[:maxlen]
        attention_mask = (attention_mask + [pad_id] * maxlen)[:maxlen]
        new_labels = (new_labels + [pad_id] * maxlen)[:maxlen]
        return input_ids, attention_mask, new_labels


    def tokenize_function(examples):
        # for instruction data, single data process

        if train_phase==1:
            text = examples['text']
            _single_label_masking = _single_label_masking_phase1
        else:
            text = examples['prompt'] + examples['ans']
            _single_label_masking = _single_label_masking_phase3
        text = text.rstrip()
        text = text[:-5] if text.endswith('[END]') else text
        # replace bos eos token
        text = text.replace('[END]','').replace('[BOS]','<s>').replace('[EOS]','</s>')

        ret = {}
        inputs = tokenizer(text, add_special_tokens=False, padding='max_length', 
                           max_length=max_len, truncation=True)
        ret["input_ids"] = inputs["input_ids"]
        ret["attention_mask"] = inputs["attention_mask"]
        ret["labels"] = _single_label_masking(ret["input_ids"].copy(), tokenizer.bos_token_id)

        if ret["input_ids"][-1] in {tokenizer.eos_token_id, tokenizer.pad_token_id}:
            ret["valid"] = 1
        else:
            ret["valid"] = 0

        return ret


    def tokenize_function_alpaca(examples):
        # !!! for alpaca format only !!!

        if train_phase==1:
            text = examples['text']
        else:
            raise Exception('not working for phase 3')
        text = text.rstrip()
        # -- alpaca format
        # system = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        # text = system + text\
        #     .replace('[END]','')\
        #     .replace('[EOS]','</s>[EOS]')\
        #     .replace('[EOS]\n[BOS]', '[EOS]\n\n[BOS]')\
        #     .replace('[BOS]','').replace('[EOS]','')\
        #     .replace('[Human]\n','### Instruction:\n<s>')\
        #     .replace('[AI]\n','### Response:\n<s>')

        # -- vicuna format
        # system = "A chat between a user and an artificial intelligence assistant. \n" + \
        #     "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        # text = system + text\
        #     .replace('[END]','')\
        #     .replace('[EOS]','</s>[EOS]')\
        #     .replace('[EOS]\n[BOS]', '[EOS]\n\n[BOS]')\
        #     .replace('[BOS]','').replace('[EOS]','')\
        #     .replace('[Human]\n','USER:\n<s>')\
        #     .replace('[AI]\n','ASSISTANT:\n<s>')

        # -- xiaoice format
        # system = "" #"You are a helpful AI assistant.\n"
        # text = system + text\
        #     .replace('[END]','')\
        #     .replace('[EOS]','</s>[EOS]')\
        #     .replace('[BOS]','').replace('[EOS]','')\
        #     .replace('[Human]\n','[Human]\n<s>')\
        #     .replace('[AI]\n','[AI]\n<s>')\
        #     .replace('[HM]:','[Human]\n<s>')\
        #     .replace('[AI]:','[AI]\n<s>')

        system = ""  # 把prefix也mask吧，因为有些数据没有prefix
        text = system + text\
                .replace('[END]','')\
                .replace('[EOS]','</s>')\
                .replace('[BOS]','<s>')\
                .replace('[Human]\n','USER:\n')\
                .replace('[AI]\n','ASSISTANT:\n')\
                .replace('[HM]:','USER:\n')\
                .replace('[AI]:','ASSISTANT:\n')
        
        ret = {}
        inputs = tokenizer(text, add_special_tokens=False)
        input_ids, attention_mask, labels = _single_label_masking_alpaca(inputs["input_ids"], 
            tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, max_len)
        ret['input_ids'] = input_ids
        ret['attention_mask'] = attention_mask
        ret['labels'] = labels

        if ret["input_ids"][-1] in {tokenizer.eos_token_id, tokenizer.pad_token_id}:
            ret["valid"] = 1
        else:
            ret["valid"] = 0

        return ret
    
    # do tokenize
    if streaming:
         # streaming 没有num_proc
        unsupervised_raw_datasets = unsupervised_raw_datasets.map(tokenize_function, remove_columns=data_columns)
        unsupervised_raw_datasets = unsupervised_raw_datasets.filter(lambda x: True if x["valid"] == 1 else False)
    else:
        unsupervised_raw_datasets = unsupervised_raw_datasets.map(tokenize_function, remove_columns=data_columns, num_proc=10)
        unsupervised_raw_datasets = unsupervised_raw_datasets.filter(lambda x: True if x["valid"] == 1 else False, num_proc=10)
    unsupervised_raw_datasets = unsupervised_raw_datasets.remove_columns(["valid"])
    # unsupervised_raw_datasets = unsupervised_raw_datasets.shuffle(seed=42) # 反正在外面意见shuffle好，加了这个速度会变慢
    return unsupervised_raw_datasets


class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                     self.small_batch_size])
        self.free()

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []
