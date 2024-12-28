
def get_prompt(query, history=[], template="qwen", system_prompt="", fun_prompt=""):
    prompt = ""
    if template == "qwen":
        prompt += "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        for i, (question, answer) in enumerate(history):
            prompt += f"<|im_start|>user\n{question}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{answer}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{query}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
    elif template == "free":
        prompt += "<|im_start|>system\n{}<|im_end|>\n".format(system_prompt)
        for i, (question, answer) in enumerate(history):
            prompt += f"<|im_start|>user\n{question}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{answer}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{query}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
    elif template == "yi":
        for i, (question, answer) in enumerate(history):
            prompt += f"<|im_start|>user\n{question}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{answer}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{query}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
    elif template == "glm4":
        for i, (question, answer) in enumerate(history):
            prompt += f"[gMASK]<sop><|user|>\n{question}"
            prompt += f"<|assistant|>\n{answer}<|user|>\n"
        prompt += f"[gMASK]<sop><|user|>\n{query}"
        prompt += f"<|assistant|>\n"
    elif template == "llama3":
        prompt +=  "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
        for i, (question, answer) in enumerate(history):
            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>\n"
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>"
        prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif template == "llama3.1":
        system_prompt_default = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible."
        if system_prompt == "":
            system_prompt = system_prompt_default
        if fun_prompt != "":
            slot = "\n"
            messages = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt + slot + fun_prompt}<|eot_id|>"]
        else:
            messages = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"]
        for i, (question, answer) in enumerate(history):
            messages.append(f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>")
            messages.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>")
        messages.append(f"<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>")
        messages.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n")
        return ''.join(messages)
    
    elif template == "funcal":
        assert fun_prompt != "", "funcal template need system"
        # system_prompt = "You are a helpful assistant.\n\n"
        if system_prompt == "":
            system_prompt = "You are a helpful assistant.\n\n"
        else:
            system_prompt = system_prompt
        sys_prompt = f"<|im_start|>system\n{system_prompt + fun_prompt}<|im_end|>\n"
        messages = [sys_prompt]
        for i, (question, answer) in enumerate(history):
            messages.append(f"<|im_start|>user\n{question}<|im_end|>\n")
            messages.append(f"<|im_start|>assistant\n{answer}<|im_end|>\n")
        messages.append(f"<|im_start|>user\n{query}<|im_end|>\n")
        messages.append(f"<|im_start|>assistant\n")
        return ''.join(messages)
    
    else:
        NotImplementedError
        
    return prompt