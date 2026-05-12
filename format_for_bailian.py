import json
import os

input_file = r"Data\score_driven_intent_train.jsonl"
output_file = r"Data\bailian_causal_train_5000.jsonl"

print("🚀 正在转换为阿里云百炼微调专属格式...")

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        data = json.loads(line.strip())
        # 转换为百炼严格要求的 messages 数组格式
        bailian_format = {
            "messages":[
                {"role": "system", "content": data["instruction"]},
                {"role": "user", "content": data["input"]},
                {"role": "assistant", "content": data["output"]}
            ]
        }
        f_out.write(json.dumps(bailian_format, ensure_ascii=False) + '\n')

print(f"✅ 转换完美成功！请去阿里云上传这个文件: {output_file}")