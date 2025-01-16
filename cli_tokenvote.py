import re
import torch
import argparse
import jsonlines
import numpy as np
import datasets
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from vote import TokenVotingModel

import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(tokens[raw_text_len:])
        # sent = sent.split("<|endoftext|>")[0]
        # sent = sent.split("\n\n\n")[0]
        # sent = sent.split("\n\n")[0]
        # sent = sent.split("Question:")[0]
        sents.append(sent)
    return sents


if __name__ == "__main__":

    # model_name_list = ['unsloth/Llama-3.2-1B-Instruct', 'pankajmathur/orca_mini_v9_5_1B-Instruct', 'KingNish/Reasoning-Llama-1b-v0.1']
    # model_name_list = ['unsloth/Llama-3.2-1B-Instruct']
    # model_name_list = ['unsloth/Llama-3.2-1B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct', 'KingNish/Reasoning-Llama-1b-v0.1']
    model_name_list = ['Qwen/Qwen2.5-0.5B-Instruct', 'pankajmathur/orca_mini_v9_5_1B-Instruct', 'KingNish/Reasoning-Llama-1b-v0.1']
    model = TokenVotingModel(model_name_list)
    model.generation_config.do_sample = False
    model.generation_config.max_new_tokens = 256

    prompt = ''
    print('>> ', end='')
    f_output = jsonlines.Writer(open('./log.jsonl', "w", encoding="utf-8"))
    while True: 
        doc = {}
        prompt = input()
        if prompt == 'exit':
            break
        doc["input"] = prompt
        messages = [
        {"role": "system", "content": "You are a chat robot"},
        {"role": "user", "content": prompt},
        ]
        completion = model.generate(prompt)
        # completion = model.generate(prompt)
        print(completion)
        doc["completion"] = completion
        f_output.write(doc)
        print('>> ', end='')

    f_output.close()