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
    last_number = re.findall(r"\d+", answer)[-1]
    return last_number

def get_response(chats, generator): 
        gen_text = generator(chats, max_new_tokens=1024)[0]  # First return sequence
        return gen_text['generated_text'][-1]['content']

def get_batch_response(chats, generator): 
    gen_texts = generator(chats, max_new_tokens=1024, batch_size=len(chats))  # First return sequence
    return [gen_text[0]['generated_text'][-1]['content'] for gen_text in gen_texts]

def prepare(model_name):
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

    config = datasets.DownloadConfig(resume_download=True, max_retries=100)
    dataset = load_dataset("gsm8k", "main", download_config=config)
    train_dataset = dataset["train"]
    train_data = [{"question": row["question"], "answer": row["answer"]} for row in train_dataset]
    test_data = dataset["test"]
    return train_data, test_data, generator

def run_once(train_data, qna, generator):
    messages = nshot_chats(nshot_data=train_data, n=N_SHOT, question=qna['question'])
    response = get_response(messages, generator)
    
    return messages, response
    
def batch_run_once(train_data, qnas, generator):
    messages = [nshot_chats(train_data, N_SHOT, qnas['question'][i]) for i in range(len(qnas))]
    response = get_batch_response(messages, generator)
    
    return messages, response

def run_all(train_data, test_data, generator, log_file_path=None, attempts=1, batch_size=4):
    correct = 0
    total = 0
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch_qna = test_data[i:i + batch_size]
        all_pred_ans_list = []
        for _ in range(attempts):
            _, responses = batch_run_once(train_data, batch_qna, generator)
            pred_ans = [extract_ans_from_response(response) for response in responses]
            all_pred_ans_list.append(pred_ans) # shape: [attempts, batch_size] 
        
        for b in range(len(batch_qna)):
            pred_ans_list = [attempts_ans_list[b] for attempts_ans_list in all_pred_ans_list]
            pred_ans = max(set(pred_ans_list), key=pred_ans_list.count)
            true_ans = extract_ans_from_response(batch_qna['answer'][b])
            total += 1
            
            if pred_ans != true_ans and log_file_path:
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    # log_file.write(f"{messages}\n\n")
                    # log_file.write(f"Response: {response}\n\n")
                    log_file.write(f"Prediction List: {pred_ans_list}\n\n")
                    log_file.write(f"Prediction: {pred_ans}\n\n")
                    log_file.write(f"Ground Truth: {batch_qna['answer'][b]}\n\n")
                    log_file.write(f"Current Accuracy: {correct/total:.3f}\n\n")
                    log_file.write('\n\n')
            else:
                correct += 1
    print(f"Final Accuracy: {correct/total:.3f}")

def run(model_name, log_dir=None, attempts=1, batch_size=9):
    log_file_path = None
    if log_dir:
        file_name = f'errors_{attempts}attempt.txt'
        log_file_path = log_dir+file_name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(log_file_path, 'w') as log_file:
            log_file.write('')
    train_data, test_data, generator = prepare(model_name)
    run_all(train_data, test_data, generator, log_file_path, attempts, batch_size)
    
if __name__ == "__main__":
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    run(model_name, 'log/llama3/', attempts=9, batch_size=9)
