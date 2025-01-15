import sys
import os
import time
sys.path.append("/Users/a58/Downloads/llm_eval/scripts")
from evaluators.qwen import LLaMA_Evaluator 
from typing import Tuple

import pandas as pd


choices = ["A", "B", "C", "D"]

generate_args = {
    "max_gen_len": 256,
    "temperature": 0.8,
    "top_p": 0.95,
}




def run_ceval(
        model_name: str = "qwen-7b-chat",
        ntrain: int = 5,
        few_shot: bool = False,
        cot: bool = False,
        subject: str = "operating_system",
        data_path: str = "",
        model = None,
        chat_model: str = "ins"
):
    evaluator = LLaMA_Evaluator(
        choices=choices,
        k=ntrain,
    )

    subject_name = subject
    run_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    save_result_dir = os.path.join(
        r"logs", f"{model_name}{'_CoT' if cot else ''}_{run_date}")
    os.makedirs(save_result_dir, exist_ok=True)
    
    
    val_file_path = data_path + os.path.join('/val', f'{subject_name}_val.csv')
    val_df = pd.read_csv(val_file_path)

    if few_shot:
        dev_file_path = data_path + os.path.join('/dev', f'{subject_name}_dev.csv')
        dev_df = pd.read_csv(dev_file_path)
        correct_ratio = evaluator.eval_subject(
            model_name,
            chat_model,
            subject_name,
            val_df,
            dev_df,
            few_shot=few_shot,
            save_result_dir=save_result_dir,
            cot=cot,
            model=model,
            **generate_args
        )
    else:
        correct_ratio = evaluator.eval_subject(
            model_name,
            chat_model,
            subject_name,
            val_df,
            save_result_dir=save_result_dir,
            cot=cot,
            model=model,
            **generate_args
        )
    return correct_ratio


if __name__ == "__main__":
    data_path = "/Volumes/save_data 1/llm_evaluate/ceval"
    files = os.listdir(data_path + "/val")
    subjects = [file.split(".")[0].strip("_val") for file in files]
    run_ceval(model_name = "qwen",
        ntrain = 5,
        few_shot = True,
        cot = True, #False表示使用模型的输出向量
        subject = "operating_system",
        data_path = data_path,
        model="/workspace/Qwen2.5-3B-Instruct")
