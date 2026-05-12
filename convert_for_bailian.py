import json
import os

# 你的 1000 条数据的原始文件
input_file = r"D:\llm_agent\Data\movie_train_part2.jsonl"
# 转换后，用于上传给阿里云的文件
output_file = r"D:\llm_agent\Data\bailian_train4.jsonl"

print("开始转换格式...")
with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        data = json.loads(line.strip())
        
        # 转换为百炼平台要求的 messages 数组格式
        bailian_format = {
            "messages": [
                {"role": "system", "content": data["instruction"]},
                {"role": "user", "content": data["input"]},
                {"role": "assistant", "content": data["output"]}
            ]
        }
        f_out.write(json.dumps(bailian_format, ensure_ascii=False) + '\n')

print(f"✅ 转换完美成功！请去上传: {output_file}")