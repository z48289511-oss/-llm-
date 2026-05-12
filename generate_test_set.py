import csv
import json
import time
import os
from openai import OpenAI

# ==========================================
# 1. 基础配置
# ==========================================
API_KEY = "sk-8837f8c25a9a4ce1a5fc4affc507c2bf"  # ⚠️ 替换为你的阿里云 API Key
INPUT_CSV = "Data/train_split.csv"       
OUTPUT_JSONL = "Data/test_set_200.jsonl" # 你的“期末考试卷”

# ⚠️ 核心防泄露设置
TRAIN_OFFSET = 5000  # 跳过前 5000 条（已经被大模型背过答案了）
TEST_LIMIT = 200     # 只抽取 200 条作为测试集

client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

def build_test_ground_truth(utterance, sentiment, emotion, da):
    """
    生成测试集的【标准答案(Ground Truth)】
    """
    prompt = f"""
    你是一名计算语言学与认知心理学专家。
    任务：对以下输入文本执行【基于情绪分数的意图推导】。
    
    输入文本：“{utterance}”
    基础参考：情感({sentiment})，细粒度({emotion})，意图标签({da})。
    
    请严格以 JSON 格式输出：
    {{
        "Step1_Emotion_Evaluation": {{
            "emotion_category": "精准的中文情绪词汇",
            "emotion_score": "情绪强烈程度(1-10的整数)",
            "emotion_reason": "简述为什么打这个分数"
        }},
        "Step2_Threshold_Analysis": {{
            "action_trigger_threshold": "触发实质性行为意图所需的临界分数(1-10的整数)"
        }},
        "Step3_Intent_Deduction": {{
            "deduced_intent": "根据前两步计算，推导出的中文核心意图",
            "intent_urgency_score": "推导出的意图紧急度(1-10的整数)",
            "deduction_equation": "推导公式与逻辑"
        }}
    }}
    只输出合法的 JSON 字符串。
    """
    try:
        response = client.chat.completions.create(
            model="qwen-max", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1 
        )
        result = response.choices[0].message.content.strip()
        if result.startswith("```json"): result = result[7:-3].strip()
        elif result.startswith("```"): result = result[3:-3].strip()
        return json.loads(result) 
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return None

# ==========================================
# 2. 生成测试集主循环
# ==========================================
print(f"🚀 启动【200条独立测试集】生成任务...")
os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

# 检查已生成的测试集进度（支持断点续传）
existing_test_count = sum(1 for _ in open(OUTPUT_JSONL, 'r', encoding='utf-8')) if os.path.exists(OUTPUT_JSONL) else 0

with open(INPUT_CSV, 'r', encoding='utf-8') as csv_file, open(OUTPUT_JSONL, 'a', encoding='utf-8') as jsonl_file:
    reader = csv.DictReader(csv_file)
    
    for i, row in enumerate(reader):
        # 1. 跳过训练集的数据（前 5000 条）
        if i < TRAIN_OFFSET:
            continue
            
        # 2. 计算当前属于测试集的第几条
        current_test_idx = i - TRAIN_OFFSET
        
        # 3. 断点续传跳过已生成的数据
        if current_test_idx < existing_test_count:
            continue
            
        # 4. 达到 200 条就停止
        if existing_test_count >= TEST_LIMIT:
            print("\n🛑 已成功生成 200 条完美测试数据，任务圆满结束！")
            break
            
        utterance = row['Utterance']
        print(f"\n[测试集 {existing_test_count+1}/{TEST_LIMIT}] 正在标注: {utterance[:20]}...")
        
        gt_json = build_test_ground_truth(utterance, row['Sentiment'], row['Emotion'], row['DA'])
        
        if gt_json:
            record = {
                "instruction": "请严格执行序列因果建模：首先评估用户的情绪分数，然后依据预设阈值，从情绪分数中数学化地推导出用户的核心意图及紧急度。",
                "input": utterance,
                "output": json.dumps(gt_json, ensure_ascii=False, indent=2)
            }
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_file.flush()
            existing_test_count += 1
            print(f"✅ 试题生成成功: 情绪[{gt_json['Step1_Emotion_Evaluation']['emotion_score']}分] -> 意图:【{gt_json['Step3_Intent_Deduction']['deduced_intent']}】")
        
        time.sleep(0.5)

print(f"\n🎉 测试集已备齐！保存在: {OUTPUT_JSONL}")