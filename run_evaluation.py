import json
import time
import os
import re
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. 评测配置
# ==========================================
TEST_FILE = "Data/test_set_200.jsonl"
OUTPUT_EXCEL = "Data/v3_experiment_results_fast.xlsx"
FINE_TUNED_MODEL_ID = os.getenv("FINE_TUNED_MODEL_ID")

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=30.0 # 强制 30 秒超时
)

def relaxed_match(raw_text, gt_word):
    if not raw_text or not gt_word: return False
    p, t = str(raw_text).lower(), str(gt_word).lower()
    for char in " 、/，。与和的了（）()[]【】":
        p = p.replace(char, "")
        t = t.replace(char, "")
    if t in p or p in t: return True
    overlap = set(t) & set(p)
    if len(overlap) >= min(len(set(t)), 2): return True
    return False

def extract_score(text, pattern):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else 0

# ==========================================
# 2. 暴力直调：跳过所有复杂代理，直接问微调模型
# ==========================================
def get_model_answer(user_input):
    system_prompt = """你是一名资深的人机交互心理建模专家。
请严格执行序列因果建模：首先评估用户的情绪分数，然后依据预设阈值，从情绪分数中数学化地推导出用户的核心意图及紧急度。
必须严格输出包含 Step1, Step2, Step3 的合法 JSON 结构。"""
    
    try:
        response = client.chat.completions.create(
            model=FINE_TUNED_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.1,
            stream=False # 🌟 关键提速：关闭流式传输，一次性拿回结果，永不卡死！
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"

# ==========================================
# 3. 极速评测主循环
# ==========================================
def run_benchmark():
    print("🚀 启动【极速版】自动化评测（关闭流式，单次直出）...")
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data =[json.loads(line.strip()) for line in f if line.strip()]

    results =[]
    for i, item in enumerate(tqdm(test_data, desc="🤖 模型答题进度")):
        user_input = item["input"]
        try:
            gt = json.loads(item["output"])
            gt_e_cat = gt.get("Step1_Emotion_Evaluation", {}).get("emotion_category", "")
            gt_i_cat = gt.get("Step3_Intent_Deduction", {}).get("deduced_intent", "")
            gt_e_score = int(gt.get("Step1_Emotion_Evaluation", {}).get("emotion_score", 0))
            gt_i_score = int(gt.get("Step3_Intent_Deduction", {}).get("intent_urgency_score", 0))
        except:
            continue
            
        start_time = time.time()
        
        # 🌟 核心：直接获取答案，没有多线程，没有循环！
        txt_report = get_model_answer(user_input)
        latency = time.time() - start_time
        
        if "ERROR:" in txt_report:
            print(f"\n⚠️ 第 {i+1} 题调用失败: {txt_report}")
            pred_e_score, pred_i_score = 0, 0
            emotion_matched, intent_matched = 0, 0
            format_valid = 0
        else:
            raw_text = txt_report.replace(" ", "").replace("\n", "")
            emotion_matched = 1 if relaxed_match(raw_text, gt_e_cat) else 0
            intent_matched = 1 if relaxed_match(raw_text, gt_i_cat) else 0
            pred_e_score = extract_score(txt_report, r'"emotion_score"\s*:\s*(\d+)')
            pred_i_score = extract_score(txt_report, r'"intent_urgency_score"\s*:\s*(\d+)')
            format_valid = 1 if pred_e_score > 0 and pred_i_score > 0 else 0

        res = {
            "Test_ID": i + 1, "Latency(s)": round(latency, 2),
            "GT_Emotion": gt_e_cat, "Emotion_Match": emotion_matched,
            "GT_Intent": gt_i_cat, "Intent_Match": intent_matched,
            "GT_E_Score": gt_e_score, "Pred_E_Score": pred_e_score,
            "GT_I_Score": gt_i_score, "Pred_I_Score": pred_i_score,
            "Format_Valid": format_valid
        }
        results.append(res)
        time.sleep(0.5)

    # 生成报表
    df = pd.DataFrame(results)
    emo_acc = df["Emotion_Match"].mean()
    int_acc = df["Intent_Match"].mean()
    valid_df = df[df["Format_Valid"] == 1]
    mae_emo = mean_absolute_error(valid_df["GT_E_Score"], valid_df["Pred_E_Score"]) if not valid_df.empty else 0
    mae_int = mean_absolute_error(valid_df["GT_I_Score"], valid_df["Pred_I_Score"]) if not valid_df.empty else 0

    print("\n\n" + "═"*60)
    print("🏆 序列因果建模系统 (V3 - 极速直出版) · 自动化评测战报 🏆")
    print("═"*60)
    print(f"🔹 测试集总样本数      : {len(df)} 条")
    print(f"🔹 格式遵循成功率      : {df['Format_Valid'].mean()*100:.2f}%")
    print(f"🔹 情感分类准确率      : {emo_acc*100:.2f}% (柔性语义匹配)")
    print(f"🔹 意图推导准确率      : {int_acc*100:.2f}% (柔性语义匹配)")
    print(f"📉 情感强度绝对误差(MAE): {mae_emo:.3f} 分")
    print(f"📉 意图紧急度误差 (MAE): {mae_int:.3f} 分")
    print(f"⏱️ 平均单次推理耗时    : {df['Latency(s)'].mean():.2f} 秒")
    print("═"*60)
    df.to_excel(OUTPUT_EXCEL, index=False)

if __name__ == "__main__":
    run_benchmark()