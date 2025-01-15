import os
from typing import Tuple
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor
from eval_ceval import run_ceval

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--eval_data", type=str, choices=["ceval","cmmlu"],default="ceval", help="需要评估的数据集")
    args.add_argument("--chat", type=bool, default=True, help="chat模型还是base模型")
    args.add_argument("--model_path", type=str, default="/workspace", help="模型路径")
    return args.parse_args()

if __name__=="__main__":
    args = get_args()
    if args.eval_data == "ceval":
        data_path = "/Volumes/save_data 1/llm_evaluate/ceval"
        files = os.listdir(data_path + "/val")
        subjects = [file.split(".")[0].strip("_val") for file in files]
        for sub in subjects:
            run_ceval(model_name = "qwen",
                ntrain = 5,
                few_shot = True,
                cot = True, #False表示使用模型的输出向量
                subject = sub,
                data_path = data_path,
                model=args.model_path,
                chat_model=args.chat)
    elif args.eval_data == "cmmlu":
        pass

