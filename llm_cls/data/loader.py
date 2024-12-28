from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from .template import get_prompt

def label_2_idx(datas):
    label2idx = {}
    label_nums = 0
    for data in datas:
        label = data["output"]
        if label not in label2idx:
            label2idx[label] = label_nums
            label_nums += 1
    idx2label = {v:k for k, v in label2idx.items()}
    return label2idx, idx2label

def get_dataset(tokenizer, model_args, data_args, training_args):
    """
    加载并预处理数据集，并返回字典格式的数据。
    
    Args:
        template: 数据模板（可自定义）。
        model_args: 模型相关参数，包含模型名称等信息。
        data_args: 数据相关参数，如数据路径、最大长度等。
        training_args: 训练相关参数，如批量大小等。
    
    Returns:
        dict: 包含训练和验证数据集的字典，适用于 Hugging Face Trainer。
    """
    # 加载数据集
    if data_args.train_dataset.endswith(".jsonl") or data_args.train_dataset.endswith(".json"):
        train_data = load_dataset("json", data_files=data_args.train_dataset, split="train")
    else:
        train_data = load_dataset("text", data_files=data_args.train_dataset, split="train")

    if data_args.eval_dataset.endswith(".jsonl") or data_args.eval_dataset.endswith(".json"):
        valid_data = load_dataset("json", data_files=data_args.eval_dataset, split="train")
    else:
        valid_data = load_dataset("text", data_files=data_args.eval_dataset, split="train")

    label2idx, idx2label = label_2_idx(train_data)
    # 数据预处理函数
    def preprocess_function(examples):
        """
        使用模板格式化数据并分词。
        """
        instructions = examples["instruction"]
        # 逐条调用 get_prompt
        texts = [get_prompt(query=inst, template="qwen") for inst in instructions]
        # 调用 tokenizer 做分词
        result = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=data_args.cutoff_len,
        )

        # 如果原始数据中有 "outputs" 字段，就在返回的字典中加一列 "labels"
        if "output" in examples:
            result["labels"] = [label2idx[label] for label in examples["output"]]
            
        return result

    # 应用预处理
    train_dataset = train_data.map(preprocess_function, batched=True)
    valid_dataset = valid_data.map(preprocess_function, batched=True)

    # 转换为 PyTorch 格式
    train_dataset = train_dataset.with_format("torch")
    valid_dataset = valid_dataset.with_format("torch")

    # 返回数据集模块
    return {
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "num_labels": len(label2idx),
        "label2idx": label2idx,
        "idx2label": idx2label
    }
