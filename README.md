# SLM_ensemble

## 数据获取方式
运行文件时，会自动从Huggingface官网下载GSM8K数据集，并划分训练集与测试集。
## 存放路径
运行结果存放在 [log](./log/) 文件夹中。
## 处理方式
运行完成后会自动输出最终正确率（Accuracy）。运行结果中存储了每一条问题的回答结果。

## Low-level 集成
### Logits level

```sh
python logit_ensemble.py
```
使用上述代码来快速启动测试（默认为使用 performance weighted 加权）。
调整 `logit_ensemble.py` 的第 69 - 76 行来改变 Logits level 的集成方法。

### Token level
- **不同模型相同输入**
```sh
python token_ensemble_multimodel.py
```
使用上述代码来快速启动测试（默认为使用所有四个模型）。
调整 `token_ensemble_multimodel.py` 的第 55 - 58 行来改变使用的模型类型。

- **相同模型不同 few-shot**
```sh
python token_ensemble_difshot.py
```
使用上述代码来快速启动测试（默认为使用五个Qwen 0.5B模型）。
调整 `token_ensemble_difshot.py` 的第 59, 60 行来改变使用的模型类型和数量。

## Answer-Level集成
对于【Answer-level集成算法1】，在命令行中运行 `python answer_ensemble_1.py`。可指定如下参数：
- `--models`: 使用的模型，以空格分隔。本代码中测试了 `Qwen/Qwen2.5-0.5B-Instruct, unsloth/Llama-3.2-1B-Instruct, tiiuae/Falcon3-1B-Instruct` 三种模型。然而该代码也可适配其他模型，不过未实验过。
- `--file_name`: 日志文件名，`'log1.txt'`
- `--log_dir`: 日志文件路径，默认值为`'log/multi_model/'`
- `--attempts`: 每个模型的query次数，默认值为1

对于【Answer-level集成算法2】，在命令行中运行 `python answer_ensemble_1.py`。可指定如下参数：
- `--models`: 使用的模型名，以空格分隔。本代码中测试了 `Qwen/Qwen2.5-0.5B-Instruct, unsloth/Llama-3.2-1B-Instruct, tiiuae/Falcon3-1B-Instruct` 三种模型。然而该代码也可适配其他模型，不过未实验过。
- `--file_name`: 日志文件名，`'log2.txt'`
- `--log_dir`: 日志文件路径，默认值为`'log/multi_model/'`
- `--attempts`: 每个模型的query次数，默认值为1
- `--round_2_model_index`: 你在第二轮投票中希望使用的模型序号，以空格分隔。

示例：
```
python answer_ensemble_1.py --models unsloth/Llama-3.2-1B-Instruct tiiuae/Falcon3-1B-Instruct Qwen/Qwen2.5-0.5B-Instruct --file_name log1.txt --log_dir log/multi_model/ --attempts 1

python answer_ensemble_2.py --models unsloth/Llama-3.2-1B-Instruct tiiuae/Falcon3-1B-Instruct Qwen/Qwen2.5-0.5B-Instruct --file_name log1.txt --log_dir log/multi_model/ --attempts 1 --round_2_model_index 0
```

## Agent-Level集成

### 运行方式与参数说明
直接在命令行中运行
```sh
python agent_ensemble.py
```
可指定如下参数：
- `--sample-output-file`: 输出文件路径，用于存储运行结果。默认值：`./log/gsm8k_res_agent_ensemble.jsonl`；
- `--planner-model`, `--solver-model`, `--evaluator-model`: 三种角色的模型指定。需满足 transformers 库的支持，并能在 Huggingface 上下载。
本代码中使用了 `Qwen/Qwen2.5-0.5B-Instruct, unsloth/Llama-3.2-1B-Instruct, tiiuae/Falcon3-1B-Instruct` 三种模型。
- `prompt-folder`: 所使用的 few-shot prompt 的存储路径。默认值为：`./data/gsm8k/`。

**注**：本文件会加载四个模型（Planner + Solver + 2 Evaluator），默认会加载在一张GPU上（可以部署在显存 24G 的 RTX3090），但可能面临资源竞争、带宽限制等问题导致运行速度较慢。因此如果具备多卡资源，建议手动在代码中将模型加载至不同GPU。
