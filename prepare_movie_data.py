import csv
import json
import time
import os
from openai import OpenAI

# ==========================================
# ⚙️ 1. 基础配置
# ==========================================
API_KEY = "sk-8837f8c25a9a4ce1a5fc4affc507c2bf"  # 你的阿里云 API Key
BASE_DIR = r"D:\llm_agent" 
INPUT_CSV = os.path.join(BASE_DIR, "Data", "train_split.csv")

# 【安全设计】：把新跑的数据存到一个新文件里，防止覆盖你之前的心血！
OUTPUT_JSONL = os.path.join(BASE_DIR, "Data", "movie_train_part2.jsonl")

# 【核心参数】：控制从哪条开始，跑多少条
START_INDEX = 1000       # 索引从0开始，1000代表跳过前1000条，从第1001条开始
PROCESS_COUNT = 5000     # 这一次你打算再跑多少条？(这里设置再跑1000条)

client = OpenAI(
    api_key=API_KEY, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def augment_data_with_llm(utterance, sentiment, emotion, da):
    """调用 Qwen-Max 进行五维标签的知识蒸馏"""
    prompt = f"""
    你是一名资深的心理与行为分析专家。
    已知用户台词：“{utterance}”
    已有基础标注：情感倾向为{sentiment}，情感分类为{emotion}，意图为{da}。
    
    请将上述基础标注翻译为中文，并评估其【情感强度】(1-10分)，最后撰写一段【理由】(解释触发该情绪的原因及意图逻辑)。
    
    请严格按以下格式输出：
    【情感倾向】：
    【情感分类】：
    【情感强度】：
    【意图识别】：
    【理由】：
    """
    try:
        response = client.chat.completions.create(
            model="qwen-max", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        return None

# ==========================================
# 🚀 2. 执行断点续传任务
# ==========================================
print(f"🚀 准备从第 {START_INDEX + 1} 条开始读取数据...")
print(f"📂 产出将追加保存至: {OUTPUT_JSONL}")

# 使用 'a' (append) 模式，这样如果脚本断了，你再跑也不会清空里面的内容
with open(INPUT_CSV, 'r', encoding='utf-8') as csv_file, open(OUTPUT_JSONL, 'a', encoding='utf-8') as jsonl_file:
    reader = csv.DictReader(csv_file)
    
    processed_this_run = 0
    
    for i, row in enumerate(reader):
        # 💡 核心逻辑：跳过前 1000 条
        if i < START_INDEX:
            continue
            
        # 💡 核心逻辑：达到本次计划处理的数量后停止
        if processed_this_run >= PROCESS_COUNT:
            print(f"\n✋ 已完成本次计划的 {PROCESS_COUNT} 条处理，任务结束。")
            break
            
        try:
            utterance = row['Utterance']
            sentiment = row['Sentiment']
            emotion = row['Emotion']
            da = row['DA']
            
            print(f"正在处理第 {i+1} 条: {utterance[:15]}...")
            
            enhanced_output = augment_data_with_llm(utterance, sentiment, emotion, da)
            
            if enhanced_output:
                record = {
                    "instruction": "你是一名资深的人机交互意图与情感分析专家。请分析用户的输入，并输出结构化的情感倾向、情感分类、情感强度、意图识别及深度关联理由。",
                    "input": utterance,
                    "output": enhanced_output
                }
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                print("   ✅ 生成成功")
            
            processed_this_run += 1
            time.sleep(0.5) # API 频率控制
            
        except KeyError as e:
            print(f"❌ CSV 格式错误: {e}")
            break

print(f"\n🎉 恭喜！新增数据处理完毕，请查看 {OUTPUT_JSONL}")