import json
import random
import transformers
import torch
import datasets
from datasets import load_from_disk, load_dataset
from vote import TokenVotingModel, TokenVotingSingleModel
import re
import os; os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def load_jsonlines(file_name: str):
    with open(file_name, 'r') as f:
        return [json.loads(line) for line in f]

def nshot_chats_for_each(nshot_data: list, n: int, question: str, n_model: int):

    def question_prompt(s):
        return f'Question: {s}'

    def answer_prompt(s):
        return f'Answer: {s}'

    chats_list = [[] for _ in range(n_model)]

    random.seed(42)
    idx = 0
    for qna in random.sample(nshot_data, n * n_model):
        chats_list[idx // n].append(
            {"role": "user", "content": question_prompt(qna["question"])})
        chats_list[idx // n].append(
            {"role": "assistant", "content": answer_prompt(qna["answer"])})

        if (idx + 1) % n == 0:
            chats_list[idx // n].append({"role": "user", "content": question_prompt(question)+" Let's think step by step. At the end, you MUST write the answer as an integer after '####'."})
        idx += 1

    return chats_list

def extract_ans_from_response(answer: str, eos=None):
    # if eos:
    #     answer = answer.split(eos)[0].strip()

    # answer = answer.split('####')[-1].strip()

    # for remove_char in [',', '$', '%', 'g']:
    #     answer = answer.replace(remove_char, '')

    # try:
    #     return int(answer)
    # except ValueError:
    try:
        last_number = re.findall(r"\d+", answer)[-1]
    except:
        last_number = 1e9
    return last_number

if __name__ == "__main__":

    n_model = 5
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = TokenVotingSingleModel(model_name, n_model=n_model)

    def get_response(chats_list): 
        gen_text = model.generate_with_different_input(chats_list, max_new_tokens=1024)  # First return sequence
        return gen_text

    download_config = datasets.DownloadConfig(resume_download=True, max_retries=100)
    dataset = load_dataset("gsm8k", "main", download_config=download_config)
    train_dataset = dataset["train"]
    train_data = [{"question": row["question"], "answer": row["answer"]} for row in train_dataset]
    test_data = dataset["test"]

    N_SHOT = 8

    import os
    if not os.path.exists('log'):
        os.makedirs('log')

    log_file_path = 'log/tokenvote_difshot_Qwen_5.txt'
    with open(log_file_path, 'w') as log_file:
        log_file.write('')

    from tqdm import tqdm
    total = correct = 0
    for qna in tqdm(test_data):

        messages_list = nshot_chats_for_each(nshot_data=train_data, n=N_SHOT, question=qna['question'], n_model=n_model)
        response = get_response(messages_list)
        
        pred_ans = extract_ans_from_response(response)
        true_ans = extract_ans_from_response(qna['answer'])

        total += 1
        if pred_ans != true_ans:
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"{messages_list}\n\n")
                log_file.write(f"Response: {response}\n\n")
                log_file.write(f"Ground Truth: {qna['answer']}\n\n")
                log_file.write(f"Current Accuracy: {correct/total:.3f}\n\n")
                log_file.write('\n\n')
        else:
            correct += 1

    print(f"Total Accuracy: {correct/total:.3f}")
