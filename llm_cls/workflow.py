import math
import torch
import logging
from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForLanguageModeling
from transformers import Seq2SeqTrainingArguments, TrainerCallback
from .hparams import DataArguments, FinetuningArguments, ModelArguments, GeneratingArguments
from .model.custom_model import QwenForClassification
from .utils.logging import setup_logging
from .utils.model_freeze import freeze_parms, parms_is_freeze_print
from .data.loader import get_dataset
from .data.collator import DataCollatorForClassification
from .trainer import CustomSeq2SeqTrainer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split

def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Training started")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    base_model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                                      device_map="auto", 
                                                      torch_dtype=torch.float32, trust_remote_code=True)
    dataset_module = get_dataset(tokenizer, model_args, data_args, training_args)
    model = QwenForClassification(qwen_model=base_model, config=base_model_config, num_labels=dataset_module["num_labels"])
    freeze_parms(model)
    parms_is_freeze_print(model)
    data_collator = DataCollatorForClassification()
    
    # if training_args.do_eval:
    #     train_data, val_data = train_test_split(
    #         dataset_module["train_dataset"], 
    #         test_size=0.2, 
    #         shuffle=True,
    #         random_state=42
    #         )
    #     train_dataset=train_data
    #     eval_dataset=val_data
    # else:
    #     train_dataset=dataset_module["train_dataset"]
    #     eval_dataset=None
    #     training_args.args.eval_strategy="no"

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        my_label2idx=dataset_module["label2idx"],
        my_idx2label=dataset_module["idx2label"],
        train_dataset=dataset_module["train_dataset"],
        eval_dataset=dataset_module["eval_dataset"].train_test_split(test_size=0.2, seed=42)["train"],
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    # if training_args.do_eval:
    #     metrics = trainer.evaluate(metric_key_prefix="eval")
    #     try:
    #         perplexity = math.exp(metrics["eval_loss"])
    #     except OverflowError:
    #         perplexity = float("inf")

    #     metrics["perplexity"] = perplexity
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)
    # Predict
    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if training_args.do_predict:
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results)