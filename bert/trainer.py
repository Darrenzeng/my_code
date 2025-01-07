# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import logging
import pandas as pd
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import Seq2SeqTrainer
from torch.nn.utils.rnn import pad_sequence
from transformers.trainer_utils import PredictionOutput, EvalPrediction
from typing_extensions import override

from transformers.trainer import _is_peft_model, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from hparams import FinetuningArguments


logger = logging.getLogger(__name__)

IGNORE_INDEX=-100

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", my_label2idx, my_idx2label, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.train_datas = kwargs.get("train_dataset")
        self.eval_dataset = kwargs.get("eval_dataset") if "eval_dataset" in kwargs else None
        self.label2idx = my_label2idx 
        self.idx2label = my_idx2label
        self.finetuning_args.num_labels = len(self.label2idx)
        # self.one_hot = np.zeros((len(self.label2idx)))


    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    # def _save(self, output_dir=None, state_dict=None):
    #     """
    #     覆盖父类的 _save 方法，使其不使用 safetensors.torch.save_file。
    #     而是用 model.save_pretrained(..., safe_serialization=False) 来避免共享权重冲突
    #     """
    #     if output_dir is None:
    #         output_dir = self.args.output_dir
    #     self.model.save_pretrained(output_dir, safe_serialization=False)  # 强制用 .bin
    #     if self.tokenizer is not None:
    #         self.tokenizer.save_pretrained(output_dir)
    #     # 保存训练状态
    #     self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        
        # 如果还需要保存 optimizer / lr_scheduler 状态，也可以自行实现
        # torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        # torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput", fine_name="generated_predictions.jsonl") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, fine_name)
        logger.info(f"Saving prediction results to {output_prediction_file}")

        predicted_labels = np.argmax(predict_results.predictions, axis=1)
        predicted_labels_str = [self.idx2label[idx] for idx in predicted_labels]

        # 转化 label_ids 为真正的标签
        true_labels_str = [self.idx2label[idx] for idx in predict_results.label_ids]

        # 合并结果为可保存的格式
        results = []
        for i, (pred, true) in enumerate(zip(predicted_labels_str, true_labels_str)):
            results.append({
                "index": i,
                "predicted_label": pred,
                "true_label": true,
                "acc":1 if pred==true else 0
            })

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            json.dump(results, writer, indent=4, ensure_ascii=False)

        res_pd = pd.DataFrame(results)
        res_pd.to_csv(output_prediction_file.split(".")[0] + ".csv", index=False)
            # writer.write("\n".join(results))

    #自定义loss函数
    def my_loss_func(self, logits, labels):
        loss_fct = CrossEntropyLoss()
        #可以把num_labels加到finetuning_args中去
        loss = loss_fct(logits.view(-1, self.finetuning_args.num_labels), labels.view(-1))
        
        return loss
    
    @override
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # if self.label_smoother is not None and "labels" in inputs:
        if self.finetuning_args.bert_learn and "labels" in inputs:
            labels = inputs.pop("labels")
        elif (self.label_smoother is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)#出loss和logits
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        #没有label,就用模型自带的loss来计算, label不能用tokenizer了，而是用onehot
        if labels is not None:
            loss = self.my_loss_func(logits=outputs, labels=labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    # @override
    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ) -> "PredictionOutput":
        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs
        results = PredictionOutput
        results = super().predict(
            test_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **gen_kwargs
        )
        
        # Step 2: Add or modify the results as needed
        
        return results
    