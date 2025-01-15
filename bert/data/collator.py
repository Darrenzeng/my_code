import torch
from typing import Dict, List, Any, Optional, Union
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding

class DataCollatorForClassification:
    """
    用于分类任务的 DataCollator，支持动态 padding。
    """

    def __init__(self, tokenizer):
        """
        初始化 DataCollator。
        :param tokenizer: 用于动态 padding 的 tokenizer 实例
        """
        self.tokenizer = tokenizer
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 提取每个样本的 "input_ids" 和 "attention_mask"
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]

        # 找到当前 batch 中最长的序列长度
        max_length = max(len(ids) for ids in input_ids)

        # 动态 padding
        padded_input_ids = torch.stack([
            torch.cat([ids, torch.full((max_length - len(ids),), self.tokenizer.pad_token_id, dtype=torch.long)])
            for ids in input_ids
        ])
        padded_attention_masks = torch.stack([
            torch.cat([mask, torch.zeros(max_length - len(mask), dtype=torch.long)])
            for mask in attention_masks
        ])

        # 拼接 labels
        labels = torch.stack([f["labels"] for f in features], dim=0)

        # 返回批处理后的字典
        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks,
            "labels": labels
        }
