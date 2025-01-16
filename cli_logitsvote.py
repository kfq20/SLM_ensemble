import re
import torch
import argparse
import jsonlines
import numpy as np
import datasets
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from vote import LogtisVotingModel

import os; os.environ['CUDA_VISIBLE_DEVICES'] = '2'
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def decode(tokens_list, tokenizer):
    sents = []
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(tokens)
        sent = sent.split("<|endoftext|>")[0]
        # sent = sent.split("\n\n\n")[0]
        # sent = sent.split("\n\n")[0]
        sent = sent.split("Question:")[0]
        sents.append(sent)
    return sents


def generate_sample(model, input_txt):
    # input_ids = model.tokenizer.encode(input_txt)
    # raw_text_len = len(input_ids)
    # context_enc = torch.tensor([input_ids]).to(model.device)
    # print(f"Input text: {input_txt}\n")
    outputs_text = model.generate(input_txt, top_k=50)
    output_text = decode(outputs_text, model.tokenizer)[0]
    # print(f"\nOutput text: {output_text}\n")
    return output_text

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test Logits Voting.")
    parser.add_argument("-m", "--method", type=str, default="weighted")
    args = parser.parse_args()

    model_name_list = ['unsloth/Llama-3.2-1B-Instruct', 'pankajmathur/orca_mini_v9_5_1B-Instruct', 'KingNish/Reasoning-Llama-1b-v0.1']
    # model_name_list = ['unsloth/Llama-3.2-1B-Instruct']
    # model_name_list = ['unsloth/Llama-3.2-1B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct', 'KingNish/Reasoning-Llama-1b-v0.1']
    print("Loading model ...")
    model = LogtisVotingModel(model_name_list=model_name_list, verbose=True, method=args.method)

    # model.generation_config.max_length = 256
    model.generation_config.max_new_tokens = 256
    model.generation_config.do_sample = False
    model.generation_config.top_p = 1.0
    model.generation_config.temperature = 1.0
    # print(model.generation_config)
    # exit()
    
    prompt = ''
    print('>> ', end='')
    log_path = 'log.jsonl'
    f_output = jsonlines.Writer(open(log_path, "w", encoding="utf-8"))
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
        # completion = generate_sample(model, message)
        completion = generate_sample(model, messages)
        print('COMPLETION>> ' + completion)
        doc["completion"] = completion
        f_output.write(doc)
        print('>> ', end='')

    f_output.close()