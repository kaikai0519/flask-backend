# main.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "Qwen/Qwen1.5-1.8B-Chat"

# 讓模型可以跨檔案共用的變數
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    print("🚀 載入模型中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        trust_remote_code=True
    )
    model.config.pad_token_id = model.config.eos_token_id
    print("✅ 模型載入完成")

def generate_answer(query: str) -> str:
    if tokenizer is None or model is None:
        raise ValueError("Model not loaded. Please call load_model() first.")

    messages = [
        {
            "role": "system",
            "content": (
                ""
            )
        },
        {"role": "user", "content": query}
    ]


    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)  # You can adjust token length if needed
    return tokenizer.decode(outputs[0], skip_special_tokens=True)