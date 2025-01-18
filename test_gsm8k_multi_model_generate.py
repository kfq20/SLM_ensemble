import json
import random
import transformers
import torch
import datasets
from datasets import load_from_disk, load_dataset
import re
import os
from tqdm import tqdm

N_SHOT = 8
SEED = 42

def load_jsonlines(file_name: str):
    with open(file_name, 'r') as f:
        return [json.loads(line) for line in f]

def nshot_chats(nshot_data: list, n: int, question: str) -> dict:

    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f'Answer: {s}'

    chats = []

    random.seed(SEED)
    for qna in random.sample(nshot_data, n):
        chats.append(
            {"role": "user", "content": question_prompt(qna["question"])})
        chats.append(
            {"role": "assistant", "content": answer_prompt(qna["answer"])})

    chats.append({"role": "user", "content": question_prompt(question)+" Let's think step by step. At the end, you MUST write the answer as an integer after '####'."})

    return chats

def extract_ans_from_response(answer: str, eos=None):
    # if eos:
    #     answer = answer.split(eos)[0].strip()

    # answer = answer.split('####')[-1].strip()

    # for remove_char in [',', '$', '%', 'g']:
    #     answer = answer.replace(remove_char, '')

    # try:
    #     return int(answer)
    # except ValueError:
    last_number = re.findall(r"\d+", answer)
    if last_number:
        last_number = last_number[-1]
    else:
        last_number = 0
    return last_number

def get_response(chats, generator): 
    gen_text = generator(chats, max_new_tokens=1024)[0]  # First return sequence
    return gen_text['generated_text'][-1]['content']

def prepare(models):
    model_list = [
        transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        ) for model_name in models
    ]
    tokenizer_list = [
        transformers.AutoTokenizer.from_pretrained(
            model_name, 
        ) for model_name in models
    ]
    generator_list = [
        transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
        ) for model, tokenizer in zip(model_list, tokenizer_list)
    ]

    config = datasets.DownloadConfig(resume_download=True, max_retries=100)
    dataset = load_dataset("gsm8k", "main", download_config=config)
    train_dataset = dataset["train"]
    train_data = [{"question": row["question"], "answer": row["answer"]} for row in train_dataset]
    test_data = dataset["test"]
    return train_data, test_data, generator_list

def run_once(train_data, qna, generator):
    messages = nshot_chats(nshot_data=train_data, n=N_SHOT, question=qna['question'])
    response = get_response(messages, generator)
    
    return messages, response
    
def run_all(train_data, test_data, generators, log_file_path=None, attempts=1):
    correct = 0
    total = 0
    for qna in tqdm(test_data):
        pred_ans_list = []
        for generator in generators:
            for _ in range(attempts):
                _, response = run_once(train_data, qna, generator)
                pred_ans = extract_ans_from_response(response)
                pred_ans_list.append(pred_ans)
        
        random.shuffle(pred_ans_list)
        pred_ans = max(set(pred_ans_list), key=pred_ans_list.count)
        true_ans = extract_ans_from_response(qna['answer'])
        total += 1
        
        if pred_ans != true_ans and log_file_path:
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                # log_file.write(f"{messages}\n\n")
                # log_file.write(f"Response: {response}\n\n")
                log_file.write(f"Prediction List: {pred_ans_list}\n\n")
                log_file.write(f"Prediction: {pred_ans}\n\n")
                log_file.write(f"Ground Truth: {qna['answer']}\n\n")
                log_file.write(f"Current Accuracy: {correct/total:.3f}\n\n")
                log_file.write('\n\n')
        else:
            correct += 1
    print(f"Final Accuracy: {correct/total:.3f}")

def run(models, log_dir=None, attempts=1):
    log_file_path = None
    if log_dir:
        file_name = f'errors_{attempts}attempt_{SEED}.txt'
        log_file_path = log_dir+file_name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(log_file_path, 'w') as log_file:
            log_file.write('')
    train_data, test_data, generators = prepare(models)
    run_all(train_data, test_data, generators, log_file_path, attempts)
    
if __name__ == "__main__":
    models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "unsloth/Llama-3.2-1B-Instruct",
        "tiiuae/Falcon3-1B-Instruct",
    ]
    run(models, f'log/multi_model/', attempts=1)