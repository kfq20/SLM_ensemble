import re
import torch
import argparse
import jsonlines
import numpy as np
import datasets
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(tokens[raw_text_len:])
        sent = sent.split("<|endoftext|>")[0]
        # sent = sent.split("\n\n\n")[0]
        # sent = sent.split("\n\n")[0]
        # sent = sent.split("Question:")[0]
        sents.append(sent)
    return sents


def generate_sample(model, tokenizer, input_txt):
    input_ids = tokenizer.encode(input_txt)
    raw_text_len = len(input_ids)
    context_enc = torch.tensor([input_ids]).to(model.device)
    # print(f"Input text: {input_txt}\n")
    outputs = model.generate(context_enc, top_k=50)
    output_text = decode(outputs, tokenizer, raw_text_len)[0]
    # print(f"\nOutput text: {output_text}\n")
    return output_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="Qwen/Qwen-7B",
    )
    parser.add_argument(
        "-o", "--log-path", type=str, default="log.jsonl"
    )

    args = parser.parse_args()

    config = datasets.DownloadConfig(resume_download=True, max_retries=100)


    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True
    )

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B-Instruct', device_map="auto", trust_remote_code=True
    ).eval()

    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    model.generation_config.do_sample = False

    prompt = ''
    print('>> ', end='')
    f_output = jsonlines.Writer(open(args.log_path, "w", encoding="utf-8"))
    while True: 
        doc = {}
        prompt = input()
        if prompt == 'exit':
            break
        doc["input"] = prompt
        completion = generate_sample(model, tokenizer, prompt)
        print(completion)
        doc["completion"] = completion
        f_output.write(doc)
        print('>> ', end='')

    f_output.close()