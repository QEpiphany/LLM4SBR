from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig
from nltk.chat import Chat
import json
import csv
from tqdm import tqdm
import openpyxl
import torch
import os

#模型下载
MODEL_PATH = "../Qwen-7B-Chat"
Process_File = "tra_session_short"

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# use fp16
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cuda:4", trust_remote_code=True, fp16=True)
model.eval()

def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


json_file_path = "data/Ml-1M/"+Process_File+".json"

with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
all_raw_text = [item['prompt'] for item in data]
all_raw_text = all_raw_text

xlsx_file_path = "data/Ml-1M/"+Process_File+".xlsx"
# 创建一个
workbook = openpyxl.Workbook()

# 选择默认的工作表（第一个工作表）
sheet = workbook.active

header = ['Key', 'Value']
sheet.append(header)

# 批量写入数据
batch_size = 5000
for batch in batch_generator(all_raw_text, batch_size):
    for raw_text in tqdm(batch, total=len(batch)):
        response, history = model.chat(tokenizer, raw_text, history=None)
        print(response)

        # 将数据写入Excel文件
        sheet.append([raw_text, response])

# 保存工作簿到xlsx文件
workbook.save(xlsx_file_path)

print(f"数据已成功写入到 {xlsx_file_path}")






