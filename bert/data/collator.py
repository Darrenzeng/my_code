import torch
from typing import Dict, List, Any, Optional, Union
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding

class DataCollatorForClassification:
    """
    用于分类任务的简单 DataCollator。
    传入的每个样本都包含:
      - "input_ids": Tensor(seq_len,) 或者 (seq_len,...) 视情况而定
      - "attention": Tensor(seq_len,)  (或 "attention_mask")
      - "labels": Tensor(1,) 或 (num_labels,) 等
    全部已是 PyTorch Tensor类型。

    作用: 将这些样本堆叠为 batch，并返回一个字典：
      {
        "input_ids": (batch_size, seq_len, ...),
        "attention": (batch_size, seq_len, ...),
        "labels": (batch_size, ...)
      }
    """

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 将每条样本的各字段在第0维拼接
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features], dim=0),
            "attention_mask": torch.stack([f["attention_mask"] for f in features], dim=0),
            "labels": torch.stack([f["labels"] for f in features], dim=0),
        }
        return batch
