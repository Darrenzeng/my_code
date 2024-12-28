import torch.nn as nn

import torch
import torch.nn as nn
from transformers import AutoModel

# 假设你已经加载好了 Qwen 的预训练模型（注意兼容性和 API）
# 下面的 "QwenModel" 仅作占位示例

class QwenForClassification(nn.Module):
    def __init__(self, qwen_model, config, num_labels=2):
        super().__init__()
        self.qwen_model = qwen_model
        hidden_size = config.hidden_size  # 需与实际 Qwen 隐藏层大小对应
        self.classifier = nn.Linear(hidden_size, num_labels)
    

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 调用 Qwen 模型，并让其输出 hidden_states
        outputs = self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  
            return_dict=True            # 一般最好显式指定 return_dict=True，得到一个 CausalLMOutputWithPast
        )
        # decoder-only 模型的最后一层 hidden states 在 outputs.hidden_states[-1]
        last_hidden_state = outputs.hidden_states[-1]   # (batch_size, seq_len, hidden_size)
        
        # 这里以取 [CLS] 位（或者说第一 token）当做 pooled vector
        # 其实 GPT 模型不存在严格意义的 [CLS]，通常我们只取第 0 个 token 或者其它位置作为分类依据
        pooled_output = last_hidden_state[:, 0, :]      # (batch_size, hidden_size)
        
        # 得到分类 logits
        logits = self.classifier(pooled_output)         # (batch_size, num_labels)

        # 如果传入了 labels，则计算 loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 注意 logits 的形状是 (batch_size, num_labels)，labels 是 (batch_size,)
            # 无需 .view(-1, self.num_labels) 和 .view(-1) 也可，但某些场景下需展开以适应维度
            loss = loss_fct(logits, labels)
            return (loss, logits)
        
        return logits