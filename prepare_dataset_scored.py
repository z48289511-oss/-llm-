import csv
import json
import time
import os
from openai import OpenAI

# ==========================================
# 1. 基础配置
# ==========================================
API_KEY = "sk-8837f8c25a9a4ce1a5fc4affc507c2bf"  # ⚠️ 务必替换为你的真实 API Key
INPUT_CSV = "Data/train_split.csv"
OUTPUT_JSONL = "Data/score_driven_intent_train.jsonl"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def build_score_driven_model(utterance, sentiment, emotion, da):
    prompt = f"""
你是一名计算语言学与认知心理学专家。
任务：对以下输入文本执行【基于情绪分数的意图推导】。

输入文本："{utterance}"
基础参考：情感({sentiment})，细粒度({emotion})，原始意图参考({da})。

请严格遵循"先评估情绪分数 -> 设定阈值 -> 推导出意图"的单向因果链条，并以严格的 JSON 格式输出：
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
        "deduction_equation": "推导公式与逻辑，例如：因为 emotion_score(8) >= trigger_threshold(6)，情绪势能溢出，推导出【强烈指责与维权】的意图"
    }}
}}

注意：只输出合法的 JSON 字符串，不要包含任何 Markdown 符号（如 ```json）和其他废话。
"""

    # 增加重试机制，防止网络偶尔抖动
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="qwen-max",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            result = response.choices[0].message.content.strip()
            if result.startswith("```json"):
                result = result[7:-3].strip()
            elif result.startswith("```"):
                result = result[3:-3].strip()

            return json.loads(result)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)  # 失败后等2秒重试
                continue
            print(f"❌ 意图推导生成失败: {e}")
            return None


# ==========================================
# 2. 批量处理主循环 (严格限制15000 条)
# ==========================================
print(f"🚀 启动【由情绪分数推导意图】的特征蒸馏 (目标: 10000条)...")
os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

# 断点续传逻辑
processed_count = 0
if os.path.exists(OUTPUT_JSONL):
    with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
        processed_count = sum(1 for _ in f)

if processed_count > 0:
    print(f"📝 发现本地已有 {processed_count} 条进度，将自动跳过已处理的数据...")

with open(INPUT_CSV, 'r', encoding='utf-8') as csv_file, \
        open(OUTPUT_JSONL, 'a', encoding='utf-8') as jsonl_file:
    reader = csv.DictReader(csv_file)

    for i, row in enumerate(reader):
        # ⚠️ 核心限制：到达 5000 条自动停止
        if i >= 20000:
            print("\n🛑 已成功处理达到 20000 条限制，任务自动圆满结束！")
            break

        if i < processed_count:
            continue

        utterance = row['Utterance']
        print(f"\n[{i+1}/20000] 正在构建因果链: {utterance[:20]}...")

        deduced_json = build_score_driven_model(
            utterance, row['Sentiment'], row['Emotion'], row['DA']
        )

        if deduced_json:
            record = {
                "instruction": "请严格执行序列因果建模：首先评估用户的情绪分数，然后依据预设阈值，从情绪分数中数学化地推导出用户的核心意图及紧急度。",
                "input": utterance,
                "output": json.dumps(deduced_json, ensure_ascii=False, indent=2)
            }
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_file.flush()

            # 控制台高亮展示
            e_score = deduced_json['Step1_Emotion_Evaluation']['emotion_score']
            threshold = deduced_json['Step2_Threshold_Analysis']['action_trigger_threshold']
            intent = deduced_json['Step3_Intent_Deduction']['deduced_intent']
            print(f"✅ 推导成功: 情绪分数[{e_score}] vs 阈值[{threshold}] ➡️ 推导意图:【{intent}】")

        time.sleep(0.5)

print(f"\n🎉 分数推导意图数据集处理完成！完美微调文件保存在: {OUTPUT_JSONL}")