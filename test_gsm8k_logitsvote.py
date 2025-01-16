import json
import random
import transformers
import torch
import datasets
from datasets import load_from_disk, load_dataset
from vote import LogtisVotingModel
import re
import os; os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def load_jsonlines(file_name: str):
    with open(file_name, 'r') as f:
        return [json.loads(line) for line in f]

def nshot_chats(nshot_data: list, n: int, question: str):

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

if __name__ == "__main__":
    model_name_list = ['unsloth/Llama-3.2-1B-Instruct', 'pankajmathur/orca_mini_v9_5_1B-Instruct', 'KingNish/Reasoning-Llama-1b-v0.1']
    performance_list = [0.469, 0.333, 0.316]
    model = LogtisVotingModel(model_name_list, method='weighted', weight_list=performance_list)
    model.generation_config.do_sample = False
    model.generation_config.top_p = 1.0
    model.generation_config.temperature = 1.0

    def get_response(chats): 
        gen_token_ids = model.generate(chats, max_new_tokens=1024)  # First return sequence
        gen_text = decode(gen_token_ids, model.tokenizer)[0]
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

    log_file_path = 'log/logits_performance.txt'
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
