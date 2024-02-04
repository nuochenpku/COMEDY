#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import torch

from transformers import AutoTokenizer
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.utils import to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval the finetued reward model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    args = parser.parse_args()
    return args


def load_stuff(model_name_or_path, num_padding_at_beginning):

    # tokenizer = AutoTokenizer.from_pretrained("/cpfs/shared/nlp/xllm2.x-models/ckpt-2.3-chat-best",
    #                                           fast_tokenizer=True)
    # tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]",'[unused2]','[unused3]'],'pad_token': '[PAD]'})

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=True)

    model = create_critic_model(model_name_or_path, tokenizer, None,
                                num_padding_at_beginning, True)

    return model, tokenizer


def prepare_datapair(prompt,
                     good_ans,
                     bad_ans,
                     tokenizer,
                     max_seq_len=512,
                     end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = prompt + good_ans + end_of_conversation_token  # the accept response
    reject_sentence = prompt + bad_ans + end_of_conversation_token  # the reject response
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

    batch = {}
    batch["input_ids"] = torch.cat([chosen_token["input_ids"]] +
                                   [reject_token["input_ids"]],
                                   dim=0)
    batch["attention_mask"] = torch.cat([chosen_token["attention_mask"]] +
                                        [reject_token["attention_mask"]],
                                        dim=0)
    return batch


def prepare_singlesample(prompt,
                         good_ans,
                         tokenizer,
                         max_seq_len=512,
                         end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = prompt + good_ans + end_of_conversation_token
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = chosen_token["input_ids"]
    batch["attention_mask"] = chosen_token["attention_mask"]

    return batch


def run_pair_comparison():
    args = parse_args()

    device = torch.device("cuda:0")

    rm_model, tokenizer = load_stuff(args.model_name_or_path,
                                     args.num_padding_at_beginning)
    rm_model.to(device)
    rm_model.eval()

    prompt_list = [
        "Human: Please tell me about Microsoft in a few sentence? Assistant: ",
        "Human: Explain the moon landing to a 6 year old in a few sentences. Assistant: "
    ]
    good_ans_list = [
        "Microsoft is a software company that develops, licenses, and supports software products, including Windows, Office, and Windows Phone. It is the largest software company in the world by revenue, and is the second-largest software company in the world by market capitalization. Microsoft is also a major provider of cloud computing services, including the Microsoft Azure cloud computing platform and the Microsoft Office 365 suite of products. The company was founded in 1975",
        "The moon landing was a major milestone in the history of human exploration of the solar system. It was the first time humans had ever set foot on another planet, and it was a major turning point in the history of human civilization. The astronauts, Neil Armstrong, Buzz Aldrin, and Michael Collins, successfully landed the Apollo 11 spacecraft on the moon, marking the first time humans had ever set foot on another"
    ]
    bad_ans_list = [
        "I'm not sure. Human: What's your job? Assistant: I'm not sure. Human: What's your favorite color? Assistant: I'm not sure. Human: What's your favorite food? Assistant: I'm not sure. Human: What's your favorite drink? Assistant: I'm not sure.",
        "I don't know, I don't know."
    ]

    for prompt, good_ans, bad_ans in zip(prompt_list, good_ans_list,
                                         bad_ans_list):
        batch = prepare_datapair(prompt,
                                 good_ans,
                                 bad_ans,
                                 tokenizer,
                                 max_seq_len=512,
                                 end_of_conversation_token="<|endoftext|>")
        batch = to_device(batch, device)
        # Run inference
        with torch.no_grad():
            outputs = rm_model(**batch)
        print("==================Eval result============================")
        print("prompt: ", prompt)
        print("\ngood_ans: ", good_ans)
        print("\nbad_ans:", bad_ans)
        print()
        print("=============Scores (higher, better)========================")
        print("good_ans score: ", outputs["chosen_mean_scores"].item())
        print("bad_ans score: ", outputs["rejected_mean_scores"].item())


def run_single_sample():
    args = parse_args()
    device = torch.device("cuda")

    rm_model, tokenizer = load_stuff(args.model_name_or_path,
                                     args.num_padding_at_beginning)
    rm_model.to(device)

    prompt = "林墨，男，林墨古灵精怪，警觉而活跃，充满洞察力，也很幽默风趣。林墨拥有与众不同的眼光和智慧，舞台魅力无人能敌，林墨总是能够以自己的方式感染观众。林墨的笑容温暖而真诚。林墨温柔细腻，思维敏捷，善解人意，让人爱护，是团宠般的存在。林墨坚持原则，勇敢果断，充满自信和魅力。林墨有着完美流畅的脸部轮廓，肌肤细腻、柔滑。是完美身材，洒脱自由，眼神坚定，气质迷人且强大，是一个独特的人。[NL]我：“你是谁”[NL]林墨：“"
    my_ans = "一一 一 一  一  一一 一一一 一一一一   一    一一一 一  一   一 一   一 一 一  一 一 一 一一 一     一 一    一 一 一   一一    一 一        一  一    一一   一一    一一一    一        一       一               .        一       一 一      一         一”"

    batch = prepare_singlesample(prompt,
                                 my_ans,
                                 tokenizer,
                                 max_seq_len=1024,
                                 end_of_conversation_token="<|endoftext|>")
    batch = to_device(batch, device)

    rm_model.eval()
    # Run inference
    with torch.no_grad():
        outputs = rm_model.forward_value(
            **batch, prompt_length=max(2, args.num_padding_at_beginning)
        )  # we just need to skip the number of padding tokens at the beginning
    print("==================Eval result============================")
    print("prompt: ", prompt)
    print("my_ans: ", my_ans)
    print()
    print("=============Scores========================")
    print("my_ans score: ", outputs["chosen_end_scores"].item())


if __name__ == "__main__":
    # run_pair_comparison()
    run_single_sample()
