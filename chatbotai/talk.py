from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, AutoConfig
import torch
import csv
import train
import threading
from accelerate import init_empty_weights

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("output-medium1")

def get_tokenizer():
    return tokenizer

def get_model():
    return model

def talk(userinput):
    for step in range(1):
        new_user_input_ids = tokenizer.encode(userinput + tokenizer.eos_token, return_tensors='pt')

        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        with open('userdataset.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            format = [userinput]
            writer.writerow(format)
        
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


