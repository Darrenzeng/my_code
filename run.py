from llm_cls.hparams import ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments
from transformers import HfArgumentParser
from llm_cls.workflow import run_sft

def get_parser():
    parser = HfArgumentParser(
        (
            ModelArguments, 
            DataArguments, 
            Seq2SeqTrainingArguments, 
            FinetuningArguments, 
            GeneratingArguments
        )
        )
    model_args, data_args, training_args, finetuning_args, generating_args = parser.parse_args_into_dataclasses()
    #模型和训练
    model_args.model_name_or_path = "/Users/a58/Downloads/pretrain_model/Qwen/Qwen2.5-0.5B-Instruct"
    training_args.do_train = True
    training_args.do_eval = True
    training_args.save_strategy = "epoch"
    training_args.save_total_limit = 1
    training_args.save_only_model = True
    training_args.learning_rate = 1e-4
    training_args.num_train_epochs = 1
    training_args.ddp_timeout = 1800000
    training_args.per_device_train_batch_size = 2
    training_args.gradient_accumulation_steps = 1
    training_args.overwrite_output_dir=True
    training_args.overwrite_cache=True
    #验证
    training_args.eval_strategy = "steps"  # 按步验证
    training_args.eval_steps = 5          # 每步验证
    training_args.logging_steps = 1       # 每步记录日志
    training_args.per_device_eval_batch_size = 1
    
    #预测
    training_args.do_predict = True
    
    #其他参数
    training_args.lr_scheduler_type="cosine"
    training_args.max_grad_norm=1.0
    training_args.warmup_steps=0
    training_args.save_safetensors=False
    training_args.run_name="qingqing"
    
    #数据
    data_args.train_dataset = "/Users/a58/Downloads/my_test/LLaMA-Factory-0.9.0/data/train_sft.json"
    data_args.eval_dataset = "/Users/a58/Downloads/my_test/LLaMA-Factory-0.9.0/data/train_sft.json"
    
    return model_args, data_args, training_args, finetuning_args, generating_args

def main():
    
    
    model_args, data_args, training_args, finetuning_args, generating_args = get_parser()
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args)

if __name__ == "__main__":
    main()