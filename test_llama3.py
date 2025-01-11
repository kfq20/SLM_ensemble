import json
import random
import transformers
import torch
import datasets
from datasets import load_from_disk, load_dataset

def load_jsonlines(file_name: str):
    with open(file_name, 'r') as f:
        return [json.loads(line) for line in f]

def nshot_chats(nshot_data: list, n: int, question: str) -> dict:

    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f'Answer: {s}'

    chats = []

    random.seed(42)
    for qna in random.sample(nshot_data, n):
        chats.append(
            {"role": "user", "content": question_prompt(qna["question"])})
        chats.append(
            {"role": "assistant", "content": answer_prompt(qna["answer"])})

    chats.append({"role": "user", "content": question_prompt(question)+" Let's think step by step. At the end, you MUST write the answer as an integer after '####'."})

    return chats

def extract_ans_from_response(answer: str, eos=None):
    if eos:
        answer = answer.split(eos)[0].strip()

    answer = answer.split('####')[-1].strip()

    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')

    try:
        return int(answer)
    except ValueError:
        return answer

model_name = "unsloth/Llama-3.2-1B-Instruct"

# bnb_config = transformers.BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    # quantization_config=bnb_config,
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name, 
)

generator = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
)

def get_response(chats): 
    gen_text = generator(chats, max_new_tokens=1024)[0]  # First return sequence
    return gen_text['generated_text'][-1]['content']

config = datasets.DownloadConfig(resume_download=True, max_retries=100)
dataset = load_dataset("gsm8k", "main", download_config=config)
train_dataset = dataset["train"]
train_data = [{"question": row["question"], "answer": row["answer"]} for row in train_dataset]
test_data = dataset["test"]

N_SHOT = 8

import os
if not os.path.exists('log'):
    os.makedirs('log')

log_file_path = 'log/errors.txt'
with open(log_file_path, 'w') as log_file:
    log_file.write('')

from tqdm import tqdm
total = correct = 0
for qna in tqdm(test_data):

    messages = nshot_chats(nshot_data=train_data, n=N_SHOT, question=qna['question'])
    response = get_response(messages)
    
    pred_ans = extract_ans_from_response(response)
    true_ans = extract_ans_from_response(qna['answer'])

    total += 1
    if pred_ans != true_ans:
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{messages}\n\n")
            log_file.write(f"Response: {response}\n\n")
            log_file.write(f"Ground Truth: {qna['answer']}\n\n")
            log_file.write(f"Current Accuracy: {correct/total:.3f}\n\n")
            log_file.write('\n\n')
    else:
        correct += 1

print(f"Total Accuracy: {correct/total:.3f}")


# messages = nshot_chats(nshot_data=train_data, n=N_SHOT, question=test_data[0]['question'])  # 8-shot prompt

# response = get_response(messages)
# print(response)

# pred_ans = extract_ans_from_response(response)
# pred_ans