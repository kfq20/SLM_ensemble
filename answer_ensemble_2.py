import json
import random
import transformers
import torch
import datasets
from datasets import load_from_disk, load_dataset
import re
import os
from tqdm import tqdm
import argparse

N_SHOT = 8

def load_jsonlines(file_name: str):
    with open(file_name, 'r') as f:
        return [json.loads(line) for line in f]

def nshot_chats(nshot_data: list, n: int, question: str) -> dict:

    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f'Answer: {s}'

    chats = []

    # random.seed(42)
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

def select_best_response(response_list, qna, generator):
    """Let a model select the best answer from all generated answers."""
    # Construct a prompt that asks the model to choose the best answer
    selection_prompt = f"""
You are given a math problem and multiple solutions. You should score each solutions according to their contents, and choose the best solution.

Problem:
{qna["question"]}

Solutions:
{f'$$$$$$$$$$$$$$${chr(10)}$$$$$$$$$$$$$$$'.join([f"Solution {i+1}: {ans}" for i, ans in enumerate(response_list)])}

Please check every solution, and please give a score from 1 to 10 for every solutions. You should NOT give the same score to different solutions. You don't have to calculate the problem by yourself but only judge according to the content. Then Tell me which solution gets the highest score. \nLet's think step by step. At the end, you should write the index of the solutions with the highest score, so your last sentence MUST be one of the following three sentence: 
1. "Therefore, Solution 1 gets the highest score."
2. "Therefore, Solution 2 gets the highest score."
3. "Therefore, Solution 3 gets the highest score."
    """
    response = get_response([{"role": "user", "content": selection_prompt}], generator)
    try:
        # Extract the index of the selected solution
        selected_index = re.findall(r"\d+", response)[-1]
        if 0 < int(selected_index) <= len(response_list):
            return int(selected_index) - 1
        else:
            return -1
    except:
        # Fallback if the response is invalid
        return -1
    
def run_all(train_data, test_data, generators, log_file_path=None, attempts=1, round_2_model_index=None):
    correct = 0
    total = 0
    for qna in tqdm(test_data):
        response_list = []
        pred_ans_list = []
        for generator in generators:
            for _ in range(attempts):
                _, response = run_once(train_data, qna, generator)
                pred_ans = extract_ans_from_response(response)
                response_list.append(response)
                pred_ans_list.append(pred_ans)
        best_response_index_list = []
        response_list_twice = response_list + response_list
        if round_2_model_index:
            round_2_model = [generators[int(i)] for i in round_2_model_index]
        else:
            round_2_model = generators
        for generator in round_2_model:
            for i in range(len(response_list)):
                index = select_best_response(response_list_twice[i: i + len(response_list)], qna, generator)
                if 0 <= index < len(response_list):
                    best_response_index_list.append((index + i)%len(response_list))
        if best_response_index_list:
            best_response_index = max(set(best_response_index_list), key=best_response_index_list.count)
            pred_ans = pred_ans_list[best_response_index]
        else:
            print("No best response index found.")
            pred_ans = max(set(pred_ans_list), key=pred_ans_list.count)
        
        # pred_ans = max(set(pred_ans_list), key=pred_ans_list.count)
        true_ans = extract_ans_from_response(qna['answer'])
        total += 1
        
        if log_file_path:
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                # log_file.write(f"{messages}\n\n")
                # log_file.write(f"Response: {response}\n\n")
                log_file.write(f"Prediction List: {pred_ans_list}\n\n")
                log_file.write(f"Prediction: {pred_ans}\n\n")
                log_file.write(f"Ground Truth: {qna['answer']}\n\n")
                log_file.write(f"Current Accuracy: {correct/total:.3f}\n\n")
                log_file.write('\n\n')
        if pred_ans == true_ans:
            correct += 1
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"Final Accuracy: {correct/total:.3f}")

def run(models, file_name=None, log_dir=None, attempts=1, round_2_model_index=None):
    log_file_path = None
    if file_name and log_dir:
        log_file_path = log_dir+file_name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(log_file_path, 'w') as log_file:
            log_file.write('')
    train_data, test_data, generators = prepare(models)
    run_all(train_data, test_data, generators, log_file_path, attempts, round_2_model_index)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Run GSM8K test with multiple models')
    parser.add_argument('--models', nargs='+', help='List of models to use', required=True)
    parser.add_argument('--file_name', help='Name of the log file', default='log2.txt')
    parser.add_argument('--log_dir', help='Directory to save log files', default='log/multi_model/')
    parser.add_argument('--attempts', type=int, help='Number of attempts for each question', default=1)
    parser.add_argument('--round_2_model_index', nargs='+', help='List of models to use for round 2', required=True)
    # example: python answer_ensemble_2.py --models unsloth/Llama-3.2-1B-Instruct tiiuae/Falcon3-1B-Instruct Qwen/Qwen2.5-0.5B-Instruct --file_name log1.txt --log_dir log/multi_model/ --attempts 1 --round_2_model_index 0 1
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.models, args.file_name, args.log_dir, args.attempts, args.round_2_model_index)
