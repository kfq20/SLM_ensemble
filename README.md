# SLM_ensemble

## Agent-Level集成

### 运行方式与参数说明
直接在命令行中运行 `python agent_ensemble.py`。可指定如下参数：
- `--sample-output-file`: 输出文件路径，用于存储运行结果。默认值：`./log/gsm8k_res_agent_ensemble.jsonl`；
- `--planner-model`, `--solver-model`, `--evaluator-model`: 三种角色的模型指定。需满足 transformers 库的支持，并能在 Huggingface 上下载。
本代码中使用了 `Qwen/Qwen2.5-0.5B-Instruct, unsloth/Llama-3.2-1B-Instruct, tiiuae/Falcon3-1B-Instruct` 三种模型。
- `prompt-folder`: 所使用的 few-shot prompt 的存储路径。默认值为：`./data/gsm8k/`。

**注**：本文件会加载四个模型（Planner + Solver + 2 Evaluator），默认会加载在一张GPU上（可以部署在显存 24G 的 RTX3090），但可能面临资源竞争、带宽限制等问题导致运行速度较慢。因此如果具备多卡资源，建议手动在代码中将模型加载至不同GPU。
### 数据获取方式
运行文件时，会自动从Huggingface官网下载GSM8K数据集，并划分训练集与测试集。
### 存放路径
运行结果存放在 [agent_ensemble_results](./agent_ensemble_results/) 文件夹中。
### 处理方式
运行完成后会自动输出最终正确率（Accuracy）。运行结果中存储了每一条问题的回答结果。