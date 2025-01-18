import re
import torch
import argparse
import jsonlines
import numpy as np
import datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.generation import GenerationConfig
from test_qwen_gsm8k import *
from tqdm import tqdm

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def load_prompts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取所有内容
        content = f.read()

    # 按空行分割内容
    qa_pairs = content.strip().split('\n\n\n')

    # 将每对问答保存为一个字典列表
    chats = []
    for pair in qa_pairs:
        # 假设每个pair包含用户和助手的内容
        qna = pair.split('\n\n')
        question, answer = qna[0], qna[1]  # user和assistant的内容
        chats.append({"role": "user", "content": question})
        chats.append({"role": "assistant", "content": answer})

    return chats

def load_few_shot_examples(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def planner_generate(task, chats, generator):
    chats_copy = chats.copy()
    chats_copy.append({"role": "user", "content": f"Question: {task}\nLet's break down the problem and suggest every step to compute, without giving the specific calculation procedure\n"})
    gen_text = generator(chats_copy, max_new_tokens=256)[0]
    return gen_text['generated_text'][-1]['content']

def solver_generate(task, tips, chats, generator):
    chats_copy = chats.copy()
    chats_copy.append({"role": "user", "content": f"Question: {task}\nHere is a step-by-step plan to solve the question:\n{tips}\nNow Let's calculate step by step\n"})
    with torch.no_grad():
        gen_text = generator(chats_copy, max_new_tokens=256)[0]
    return gen_text['generated_text'][-1]['content']

def planner_critic_generate(question, tips, generator):
    message = [{"role": "user", "content": f"Evaluate the analysis of a given question: Question: {question}; Analysis: {tips}. If the logic is flawed or specific numerical calculations are performed, output 'Incorrect Answer.' Otherwise, output 'Correct Answer.'"}]
    gen_text = generator(message, max_new_tokens=256)[0]
    return gen_text['generated_text'][-1]['content']

def solver_critic_generate(question, answers, generator):
    message = [{"role": "user", "content": f"Evaluate whether the answer of a given question is correct or incorrect: Question: {question}; Answer: {answers}"}]
    gen_text = generator(message, max_new_tokens=256)[0]
    return gen_text['generated_text'][-1]['content']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="./agent_ensemble_results/gsm8k_res_agent_ensemble.jsonl"
    )
    parser.add_argument(
        "--planner-model", 
        type=str, 
        default="unsloth/Llama-3.2-1B-Instruct", 
        choices=["l", "unsloth/Llama-3.2-1B-Instruct", "f", "tiiuae/Falcon3-1B-Instruct", "q", "Qwen/Qwen2.5-0.5B-Instruct"], 
        dest="planner_model"
    )

    parser.add_argument(
        "--solver-model", 
        type=str, 
        default="Qwen/Qwen2.5-0.5B-Instruct", 
        choices=["l", "unsloth/Llama-3.2-1B-Instruct", "f", "tiiuae/Falcon3-1B-Instruct", "q", "Qwen/Qwen2.5-0.5B-Instruct"], 
        dest="solver_model"
    )

    parser.add_argument(
        "--evaluator-model", 
        type=str, 
        default="Qwen/Qwen2.5-0.5B-Instruct", 
        choices=["l", "unsloth/Llama-3.2-1B-Instruct", "f", "tiiuae/Falcon3-1B-Instruct", "q", "Qwen/Qwen2.5-0.5B-Instruct"], 
        dest="evaluator_model"
    )
    parser.add_argument("--prompt-folder", type=str, default="./data/gsm8k/")
    args = parser.parse_args()

    # Load models and tokenizers
    planner_model = AutoModelForCausalLM.from_pretrained(args.planner_model, device_map="cuda:0", trust_remote_code=True).eval()
    planner_tokenizer = AutoTokenizer.from_pretrained(args.planner_model, trust_remote_code=True)

    solver_model = AutoModelForCausalLM.from_pretrained(args.solver_model, device_map="cuda:1", trust_remote_code=True).eval()
    solver_tokenizer = AutoTokenizer.from_pretrained(args.solver_model, trust_remote_code=True)

    planner_critic_model = AutoModelForCausalLM.from_pretrained(args.evaluator_model, device_map="cuda:2", trust_remote_code=True).eval()
    planner_critic_tokenizer = AutoTokenizer.from_pretrained(args.evaluator_model, trust_remote_code=True)

    solver_critic_model = AutoModelForCausalLM.from_pretrained(args.evaluator_model, device_map="cuda:3", trust_remote_code=True).eval()
    solver_critic_tokenizer = AutoTokenizer.from_pretrained(args.evaluator_model, trust_remote_code=True)

    planner_chats = load_prompts(file_path="./data/gsm8k/planner_examples.txt")
    solver_chats = load_prompts(file_path="./data/gsm8k/solver_examples.txt")
    # planner_chats = []
    planner = pipeline(
        "text-generation",
        model=planner_model,
        tokenizer=planner_tokenizer,
        pad_token_id=planner_tokenizer.eos_token_id,
    )

    solver = pipeline(
        "text-generation",
        model=solver_model,
        tokenizer=solver_tokenizer,
        pad_token_id=solver_tokenizer.eos_token_id,
    )

    planner_critic = pipeline(
        "text-generation",
        model=planner_critic_model,
        tokenizer=planner_critic_tokenizer,
        pad_token_id=planner_critic_tokenizer.eos_token_id,
    )

    solver_critic = pipeline(
        "text-generation",
        model=solver_critic_model,
        tokenizer=solver_critic_tokenizer,
        pad_token_id=planner_tokenizer.eos_token_id,
    )
    
    # solver_chats = []

    # solver_model = AutoModelForCausalLM.from_pretrained(args.solver_model, device_map="auto", trust_remote_code=True)
    # solver_tokenizer = AutoTokenizer.from_pretrained(args.solver_model, trust_remote_code=True)

    # evaluator_model = AutoModelForCausalLM.from_pretrained(args.evaluator_model, device_map="auto", trust_remote_code=True).eval()
    # evaluator_model.generation_config = GenerationConfig.from_pretrained(
    #     args.checkpoint_path, trust_remote_code=True
    # )
    # evaluator_model.generation_config.max_length = 2048
    # evaluator_model.generation_config.do_sample = False
    # evaluator_tokenizer = AutoTokenizer.from_pretrained(args.evaluator_model, trust_remote_code=True)

    solver_examples = load_few_shot_examples(args.prompt_folder + "solver_examples.txt")
    # evaluator_examples = load_few_shot_examples(args.prompt_folder + "evaluator_examples.txt")

    config = datasets.DownloadConfig(resume_download=True, max_retries=100)
    dataset = load_dataset("gsm8k", "main", download_config=config)
    
    test = dataset["test"]

    f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))
    tot_length = test.num_rows
    acc_res = []

    max_plan_counts = 5
    max_solve_counts = 5

    error_words = ['incorrect', 'error', 'flaw']
    for doc in tqdm(test):
        task = doc["question"]
        print(f"\n\nQuestion:\n{task}\n\n")
        known_conditions = "None"
        plan_count = 1
        final_answer = None

        while True:
            # Use the planner to generate the next step
            planner_output = planner_generate(task, planner_chats, planner)
            tips = planner_output.strip()
            # print(f"\n====== planner count: {plan_count} ======\n\n{tips}")
            planner_correctness = planner_critic_generate(question=task, tips=tips, generator=planner_critic)
            # print(f"\n------ planer critic: ------ \n\n {planner_correctness}\n")
            if any(keyword in planner_correctness.lower() for keyword in error_words):
                if plan_count > max_plan_counts:
                    break
                plan_count += 1
                continue

            else:
                solve_counts = 1
                while True:
                    completion = solver_generate(task, tips, solver_chats, solver).replace(",", "")
                    # print(f"\n AAAAAAAAAAA \n\n\nsolver input {solver_chats}")
                    # print(f"\n ||||||| [Solver count: {solve_counts} |||||| \n\n{completion}")
                    solver_correctness = solver_critic_generate(task, completion, solver_critic).replace(",", "")
                    # print(f"\n------ Solver critic: ------ \n\n {solver_correctness}\n")
                    solve_numbers = re.findall(r"\d+", completion)
                    if solve_numbers:
                        last_solver_number = solve_numbers[-1]
                    else:
                        last_solver_number = 0
                    critic_numbers = re.findall(r"\d+", solver_correctness)
                    if critic_numbers:
                        last_critic_number = critic_numbers[-1]
                    else:
                        last_critic_number = 0
                    if last_solver_number != last_critic_number:
                        solve_counts += 1
                        if solve_counts > max_solve_counts:
                            break
                        continue
                    else:
                        break
                torch.cuda.empty_cache()
                doc['plan_num'] = plan_count
                doc['solve_num'] = solve_counts
                break
        answer = doc["answer"]
        acc = is_correct(completion, answer)
        print(f"\n\nAnswer: {acc}\n\n")
        doc["completion"] = completion
        doc["planner"] = tips
        doc["acc"] = acc
        doc["critic"] = solver_correctness
        f_output.write(doc)
        acc_res.append(acc)

    f_output.close()
    print("Acc: ", np.mean(acc_res))