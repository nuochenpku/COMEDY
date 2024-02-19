import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,6'
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm 
import csv
import os
import json
import os

# llama2-compressive-memory

ckpt = "/output/compress_memory/13b_v2/2023-12-28-00.12.51/"
print("loading ckpt: ", ckpt)
# device = torch.device('cuda')
model = LlamaForCausalLM.from_pretrained(ckpt, device_map='auto')
print(model.config)
model.tie_weights()  # 添加这一行来绑定模型权重
tokenizer = AutoTokenizer.from_pretrained(ckpt)
# Ensure the model is on the GPU

print("loaded ckpt: ", ckpt)



task1_prompt = "<s>[Human]\n任务名称: 双人对话的记忆描述生成。任务需求: 基于两个人的对话内容，创建两个人的客观记忆描述，以【xxx|xxx|xxx】的格式表示，其中每个xxx都是一条记忆。记忆需以对话人的姓名为主语，所有相关的对话内容都不能遗漏，不同记忆之间用'|'分割。对话内容是: \n{history}</s>\n<s>[AI]\n"
task2_prompt = "<s>[Human]\n这是一项关于定制用户描述，关系描述和事件描述的任务.需要输出的文本内容是三部分，第一部分是用户描述，主要是包括用户的信息总结。第二部分是用户和机器人的关系描述。第三部分是用户和机器人的共同事件描述。依据参考资料，提取并汇总用户的性格特征、行为模式等不同信息。需要注意的是，所有涉及用户的各个方向的信息都需要被记录并包含在用户描述中，不能有任何遗漏。最后得出一段客观的用户描述。如果参考资料违反了相关安规，涉及性、色情、暴力等内容，应直接回应：【很抱歉，相关内容涉及到了性、色情、暴力等内容，无法给出合适的输出】。用户描述要包含的用户信息包括但不限于：基本信息（如姓名、昵称、性别、外貌、生日、星座等），用户的爱好和不喜欢的事物，以及用户的各种状态（如情感状态、情绪状态、工作状态、健康状态等）。第二部分是用户和机器人的关系描述，描述对话中展现的关系亲密程度。第三部分是用户和机器人的共同事件描述，对话中发生过的事件总结。在输出描述中尽可能列举出参考资料中提及的具体例子，并保留一些有趣的信息。但请避免输出和用户无关的内容、输出的内容要少于200个字。Let's think step by step。每部分内容按照'###'进行分割。示例格式如下【用户描述：XXX###关系描述:XXX###事件描述:XXX】。输出示例如下：用户性格上特别XXX, 因为他曾经XXX，同时用户喜欢xxx，不喜欢xxx。\n用户的名称是：{user}，机器人名称：{botname} 参考资料是{memory} 输出结果是：</s>\n<s>[AI]\n"
task3_prompt = "<s>[Human]\n这个任务是综合用户压缩记忆和对话历史，给出结合压缩记忆的对话回复。用户压缩记忆是:\n {compress_memory} \n对话历史是:\n {history}\n{bot}：。\n给出合适的对话回复。</s>\n<s>[AI]\n"




def LLAMA2_13B_API_LOCAL(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)  # Ensure the input tensor is on the GPU
    # init parmeters
    generate_ids = model.generate(input_ids, max_new_tokens=2048, do_sample=True, top_k=30, top_p=0.85, temperature=0.5, repetition_penalty=1.1, eos_token_id=2, bos_token_id=1, pad_token_id=0)
    # revise parameters for memory-profile-preference
    #generate_ids = model.generate(input_ids, max_new_tokens=2048, do_sample=True, top_k=5, top_p=1, temperature=0.1, repetition_penalty=1.1, eos_token_id=2, bos_token_id=1, pad_token_id=0)
    #print(generate_ids)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    # print(output)
    response = output[len(prompt):]
    return response

def compressive_memory_infer():
    with open("/cpfs/user/chennuo/dsChatLLama/test_data/results/test_new.json", "r") as f1:
        
        datas = json.load(f1)
        
    generate_alls = {}
    for key, value in tqdm(datas.items()):
        memorys = []
        historys = []
        responses = []
        compress_memory = []
        generate_dic = {}
        for k, v in value.items():
            # for task 1 generate memory:
            
            if len(memorys) <=14:
                history = ' '.join(v['sub_history'])
                task1_example = task1_prompt.format(history=history)
                response = LLAMA2_13B_API_LOCAL(prompt=task1_example)
                response = response.replace('【','').replace('】','')
                memorys.append(response) 
                if "</s>" in response:
                    response = response.split("</s>")[0]
                
                historys.append(history)
                v['13_v2_memory'] = response.strip('I]\n')
            else:

                if len(v['sub_history'][4].split('\n')[0]) <=10:
                    continue

                if len(responses) >=5: 
                    break
                else:
                    if len(compress_memory) == 0:
                        
                        his_memory = '。'.join(memorys)
                        task2_example = task2_prompt.format(user=v['username'], botname=v['botname'], memory=his_memory)
                        response = LLAMA2_13B_API_LOCAL(prompt=task2_example)
                        response = response.replace('【','').replace('】','')
                        compress_memory.append(response.strip('I]\n'))
                        
                        if "</s>" in response:
                            response = response.split("</s>")[0]
                        v['13b_v2_compress_memory'] = response.strip('I]\n')
                        # print('compress_memory: ',response)
                
                    botname = v['botname']
                    v['sub_history'] = v['sub_history'][:5]
                    c_history = ' '.join(v['sub_history'][:-1]) + v['sub_history'][-1].split('\n')[0]
                    task3_example = task3_prompt.format(compress_memory=compress_memory[0], history=c_history, bot=botname)
                    response = LLAMA2_13B_API_LOCAL(prompt=task3_example)
                    if "</s>" in response:
                        response = response.split("</s>")[0]
                    responses.append(response.strip('I]\n'))
                    v['13b_v2_compress_memory_output'] = response.strip('I]\n')
                    # print('compress_output: ',response)
            generate_dic[k] = v
        generate_alls[key] = generate_dic

    with open('results/test_7b_v2.json','w', encoding='utf-8') as w:
        json.dump(generate_alls, w, ensure_ascii=False, indent=4)    

#Compressive Memory Infer
compressive_memory_infer()

