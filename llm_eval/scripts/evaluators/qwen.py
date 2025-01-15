import os
import sys
sys.path.append("/Users/a58/Downloads/my_test/my_code/llm_eval")
import re
from typing import List

import numpy as np
import torch
from evaluators.evaluator import Evaluator
from tqdm import tqdm
from utils.tools import get_prompt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import requests

# API endpoint
url = "http://localhost:8000/v1/completions"

# Headers
headers = {
    "Content-Type": "application/json"
}

stop = ["<|endoftext|>", "<|im_end|>",'<|im_start|>']

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class LLaMA_Evaluator(Evaluator):

    def __init__(self, choices, k=-1) -> None:
        self.choices = choices
        self.k = k
        self.patterns = [
            "答案是?\s?([ABCD])",
            "答案是?\s?：([ABCD])",
            "答案是?\s?:([ABCD])",
            "答案应该?是\s?([ABCD])",
            "答案应该?选\s?([ABCD])",
            "答案为\s?([ABCD])",
            "选择\s?([ABCD])",
            "只有选?项?\s?([ABCD])\s?是?对",
            "只有选?项?\s?([ABCD])\s?是?错",
            "只有选?项?\s?([ABCD])\s?不?正确",
            "只有选?项?\s?([ABCD])\s?错误",
            "说法不?对选?项?的?是\s?([ABCD])",
            "说法不?正确选?项?的?是\s?([ABCD])",
            "说法错误选?项?的?是\s?([ABCD])",
            "([ABCD])\s?是正确的",
            "([ABCD])\s?是正确答案",
            "选项\s?([ABCD])\s?正确",
            "所以答\s?([ABCD])",
            "1.\s?([ABCD])[.。$]?$",
            "所以\s?([ABCD][.。$]?$)",
            "所有\s?([ABCD][.。$]?$)",
            "[\s，：:,]([ABCD])[。，,\.]?$",
            "[\s，,：:][故即]([ABCD])[。\.]?$",
            "[\s，,：:]因此([ABCD])[。\.]?$",
            "[是为。]\s?([ABCD])[。\.]?$",
            "因此\s?([ABCD])[。\.]?$",
            "显然\s?([ABCD])[。\.]?$",
            "1.\s?(.*?)$",
            "答案是\s?(\S+)(?:。|$)",
            "答案应该是\s?(\S+)(?:。|$)",
            "答案为\s?(\S+)(?:。|$)",
        ]

    def format_example(self, line, include_answer=True, cot=False):
        example = line['question']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        if include_answer:
            if cot:
                example += "\n答案：让我们一步一步思考，\n" + \
                    line["explanation"] + f"\n所以答案是{line['answer']}。\n\n"
            else:
                example += '\n答案：' + line["answer"] + '\n\n'
        else:
            if cot:
                example += "\n答案：让我们一步一步思考，\n1."
            else:
                example += '\n答案：'
        return example

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += self.format_example(
                dev_df.iloc[i, :],
                include_answer=True,
                cot=cot
            )
        return prompt

    def generate(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        model: str = ""
    ) -> List[str]:
        #将prompt进行包装，然后送给vllm进行推理即可
        prompt = get_prompt(prompt, template="qwen")
        stop = ["<|endoftext|>", "<|im_end|>",'<|im_start|>']
        payload = dict(
                model = model,
                prompt=prompt,
                max_tokens=max_gen_len,
                top_p=top_p,
                top_k=40,
                temperature=temperature,
                stop=stop,
                stream=False,
                repetition_penalty=1.05,
                skip_special_tokens=False)

        # Send a POST request to the server
        out = requests.post(url, headers=headers, json=payload)
        if out.status_code == 200:
            # Extract the 'text' from the response
            response_json = out.json()
            response = response_json['choices'][0]['text']
            print("Extracted text:", response)
        else:
            # If there was an error, print the status code and content of the response
            print(f"Failed with status code {out.status_code}:")
            print(out.text)
            response = "error"
        
        return response
    
    def extract_model_answer(self,text, a,b,c,d):
        option_str=re.escape('A. '+a+'\nB. '+b+'\nC. '+c+'\nD. '+d)
        match = re.search(rf'{option_str}([\s\S]*)$', text)
        if match:
            return match.group(1)
        else:
            return None
    
    def extract_answer_option(self,text):
        match = re.findall(r'(让我们一步一步思考[\s\S]+?)(?:(?=让我们一步一步思考)|$)', text)
        text=match[0]
        regexes = [re.compile(pattern) for pattern in self.patterns]
        for regex in regexes:
            match = regex.search(text)
            if match:
                return match.group(1)
        return None
    
    def answer_str(self,answer,a,b,c,d):
        if answer=='D':
            return d
        elif answer=='C':
            return c
        elif answer=='B':
            return b
        else:
            return a

    def extract_answer(self, row, output):
        pred = {"A":0, "B":1, "C":2, "D":3}
        correct_answer_str=self.answer_str(row['answer'], row['A'],row['B'],row['C'],row['D'])
        generate_answer=self.extract_model_answer(str(output), row['A'],row['B'],row['C'],row['D'])
        model_answer=self.extract_answer_option(generate_answer)
        if row['answer']==model_answer or correct_answer_str==model_answer:
            return pred[model_answer] if model_answer in pred else model_answer, 1
        else:
            return pred[model_answer] if model_answer in pred else model_answer, 0

    def process_single_row(
        self,
        row,
        few_shot_prompt,
        model,
        model_name,
        chat_model,
        cot,
        **kwargs
    ):
        """
        多线程内部实际处理逻辑：
        1. 构造 prompt
        2. 请求服务端获取结果
        3. 根据是否是 chain-of-thought (cot) 或者普通单选题进行后处理
        """
        # 1. 构造 prompt
        question = self.format_example(row, include_answer=False, cot=cot)
        full_prompt = few_shot_prompt + question
        if chat_model:
            templat = "qwen" if "qwen" in model_name else "llama"
            full_prompt = get_prompt(query=full_prompt, template=templat)

        payload = dict(
            model=model,  # 模型路径 / 名字
            prompt=full_prompt,
            max_tokens=kwargs.get("max_gen_len", 128),
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 40),
            temperature=kwargs.get("temperature", 0.7),
            stop=stop,
            stream=False,
            repetition_penalty=1.05,
            skip_special_tokens=False,
        )

        # 2. 请求服务端获取推理结果
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            response_json = response.json()
            output = response_json["choices"][0]["text"]
        else:
            print(f"Failed with status code {response.status_code}: {response.text}")
            output = response.text

        # 3. 后处理
        if cot:
            assert isinstance(output, str)
            pred, correct = self.extract_answer(row, output)
        else:
            # 若是普通四选一题：A, B, C, D
            # 假设服务端返回的 output 是可直接 flatten() 的 logits
            logits = output.flatten()  # 具体按实际情况处理
            probs = (
                torch.nn.functional.softmax(
                    torch.tensor([
                        logits[self.tokenizer.encode("A", bos=False, eos=False)[0]],
                        logits[self.tokenizer.encode("B", bos=False, eos=False)[0]],
                        logits[self.tokenizer.encode("C", bos=False, eos=False)[0]],
                        logits[self.tokenizer.encode("D", bos=False, eos=False)[0]],
                    ]),
                    dim=0
                )
                .detach()
                .cpu()
                .numpy()
            )
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
            correct = 1 if pred == row["answer"] else 0

        return pred, correct

    def eval_subject(
        self,
        model_name,
        chat_model,
        subject_name,
        test_df,
        dev_df=None,
        few_shot=False,
        save_result_dir=None,
        cot=True,
        model="",
        max_workers = 10,
        **kwargs
    ):
        few_shot_prompt = self.generate_few_shot_prompt(
            subject_name, dev_df, cot=cot) if few_shot else []
        results = []
        scores = []

        # 创建线程池
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 将待处理的每个 row 提交到线程池
            futures = []
            for _, row in test_df.iterrows():
                future = executor.submit(
                    self.process_single_row,
                    row,
                    few_shot_prompt,
                    model,
                    model_name,
                    chat_model,
                    cot,
                    **kwargs
                )
                futures.append(future)

            # 使用 as_completed 获取已完成的任务并显示进度
            for future in tqdm(as_completed(futures), total=len(futures)):
                pred, correct = future.result()  # 获取线程返回的结果
                results.append(pred)
                scores.append(correct)

        correct_ratio = 100 * sum(scores) / len(scores)
        
        # print(f"Correct ratio: {correct_ratio:.2f}%")
        if save_result_dir:
            test_df['model_output'] = results
            test_df["correctness"] = scores
            test_df.to_csv(os.path.join(
                save_result_dir, f'{subject_name}_test.csv'), encoding="utf-8", index=False)
        return correct_ratio

    def eval_subject1(
        self,
        model_name,
        chat_model,
        subject_name,
        test_df,
        dev_df=None,
        few_shot=False,
        save_result_dir=None,
        cot=True,
        model="",
        **kwargs
    ):
        result = []
        score = []
        few_shot_prompt = self.generate_few_shot_prompt(
            subject_name, dev_df, cot=cot) if few_shot else []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=cot)
            full_prompt = few_shot_prompt + question
            if chat_model:
                templat = "qwen" if "qwen" in model_name else "llama"
                full_prompt = get_prompt(query=full_prompt, template=templat)
                
            payload = dict(
                            model = model,#模型路径名字
                            prompt=full_prompt,
                            max_tokens=kwargs.get("max_gen_len"),
                            top_p=kwargs.get("top_p"),
                            top_k=kwargs.get("top_k"),
                            temperature=kwargs.get("temperature"),
                            stop=stop,
                            #logits_processors=True,
                            # logprobs=1, #出每一个token的最高几个概率
                            stream=False,
                            repetition_penalty=1.05,
                            skip_special_tokens=False)

            # Send a POST request to the server
            response = requests.post(url, headers=headers, json=payload)

            # Check if the request was successful
            if response.status_code == 200:
                # Print out the JSON response from the server
                response_json = response.json()
                output = response_json['choices'][0]['text']
            else:
                print(f"Failed with status code {response.status_code}:")
                output = response.text
                print(response.text)
                
            # output = self.generate(
            #     full_prompt,
            #     max_gen_len=kwargs.get("max_gen_len", 512),
            #     temperature=kwargs.get("temperature", 0.8),
            #     top_p=kwargs.get("top_p", 0.95),
            #     model=model,
            # )
            if cot:
                assert isinstance(output, str)
                pred, correct = self.extract_answer(row, output)
            else:
                logits = output.flatten()
                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(
                            [
                                logits[self.tokenizer.encode(
                                    "A", bos=False, eos=False)[0]],
                                logits[self.tokenizer.encode(
                                    "B", bos=False, eos=False)[0]],
                                logits[self.tokenizer.encode(
                                    "C", bos=False, eos=False)[0]],
                                logits[self.tokenizer.encode(
                                    "D", bos=False, eos=False)[0]],
                            ]
                        ),
                        dim=0,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
                correct = 1 if pred == row['answer'] else 0
            result.append(pred)
            score.append(correct)
        correct_ratio = 100*sum(score)/len(score)

        if save_result_dir:
            test_df['model_output'] = result
            test_df["correctness"] = score
            test_df.to_csv(os.path.join(
                save_result_dir, f'{subject_name}_test.csv'), encoding="utf-8", index=False)
        return correct_ratio
