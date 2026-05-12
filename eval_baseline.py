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
OUTPUT_EXCEL = "Data/ablation_study_results.xlsx"

# 获取模型 ID (从 .env 中读取)
BASELINE_MODEL = "qwen2.5-7b-instruct"            # 原生模型 (基线)
FINETUNED_MODEL = os.getenv("FINE_TUNED_MODEL_ID") # 你的微调模型 (Model A)

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=30.0
)

# ==========================================
# 2. 柔性语义匹配 & 分数提取
# ==========================================
def relaxed_match(raw_text, gt_word):
    """柔性匹配：搜索核心词是否命中"""
    if not raw_text or not gt_word: return False
    p, t = str(raw_text).lower(), str(gt_word).lower()
    for char in " 、/，。与和的了（）()[]【】":
        p = p.replace(char, "")
        t = t.replace(char, "")
    
    if t in p or p in t: return True
    overlap = set(t) & set(p)
    if len(overlap) >= min(len(set(t)), 2): return True
    return False

def extract_score(text, field_name):
    """从大段文本中暴力抓取分数"""
    match = re.search(f'"{field_name}"\s*:\s*(\d+|"[^"]+")', text)
    if match:
        val = match.group(1).replace('"', '')
        if val.isdigit(): return int(val)
        if val in ["极高", "高", "重度"]: return 9
        if val in ["中", "中等"]: return 5
        if val in ["低", "轻微"]: return 2
    return 0  # 没抓到算作0分，让 MAE 惩罚它！

# ==========================================
# 3. 单次推理函数
# ==========================================
def get_model_answer(user_input, model_id):
    """让指定的模型去单次硬答"""
    prompt = f"""
    请对以下用户输入进行情感与意图的序列因果分析。
    用户输入：“{user_input}”
    请以 JSON 格式输出：
    {{
        "Step1_Emotion_Evaluation": {{"emotion_category": "精准的情绪词", "emotion_score": "情感强度1-10分"}},
        "Step3_Intent_Deduction": {{"deduced_intent": "核心意图", "intent_urgency_score": "意图紧急度1-10分"}}
    }}
    """
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return resp.choices[0].message.content
    except Exception as e:
        return ""

# ==========================================
# 4. 评测主引擎
# ==========================================
def run_ablation_study():
    print("🚀 启动【消融实验】自动化双轨评测 (Baseline vs Model A)...")
    
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line.strip()) for line in f if line.strip()]

    results_baseline = []
    results_model_a = []

    for i, item in enumerate(tqdm(test_data, desc="🤖 双模型同台竞技中")):
        user_input = item["input"]
        try:
            gt = json.loads(item["output"])
            gt_e_cat = gt.get("Step1_Emotion_Evaluation", {}).get("emotion_category", "")
            gt_i_cat = gt.get("Step3_Intent_Deduction", {}).get("deduced_intent", "")
            gt_e_score = int(gt.get("Step1_Emotion_Evaluation", {}).get("emotion_score", 0))
            gt_i_score = int(gt.get("Step3_Intent_Deduction", {}).get("intent_urgency_score", 0))
        except: continue
        
        # 🛡️ 模型 1：Baseline (原生 Qwen2.5-7B) 作答
        ans_base = get_model_answer(user_input, BASELINE_MODEL)
        txt_base = ans_base.replace(" ", "").replace("\n", "")
        
        res_base = {
            "Model": "Baseline",
            "Emotion_Acc": 1 if relaxed_match(txt_base, gt_e_cat) else 0,
            "Intent_Acc": 1 if relaxed_match(txt_base, gt_i_cat) else 0,
            "GT_E_Score": gt_e_score, "Pred_E_Score": extract_score(ans_base, "emotion_score"),
            "GT_I_Score": gt_i_score, "Pred_I_Score": extract_score(ans_base, "intent_urgency_score")
        }
        results_baseline.append(res_base)
        time.sleep(0.3)

        # 🛡️ 模型 2：Model A (你的专属微调模型) 作答
        ans_tuned = get_model_answer(user_input, FINETUNED_MODEL)
        txt_tuned = ans_tuned.replace(" ", "").replace("\n", "")
        
        res_tuned = {
            "Model": "Model A (LoRA Only)",
            "Emotion_Acc": 1 if relaxed_match(txt_tuned, gt_e_cat) else 0,
            "Intent_Acc": 1 if relaxed_match(txt_tuned, gt_i_cat) else 0,
            "GT_E_Score": gt_e_score, "Pred_E_Score": extract_score(ans_tuned, "emotion_score"),
            "GT_I_Score": gt_i_score, "Pred_I_Score": extract_score(ans_tuned, "intent_urgency_score")
        }
        results_model_a.append(res_tuned)
        time.sleep(0.3)

    # ==========================================
    # 5. 聚合数据并生成最终对比报表
    # ==========================================
    df_base = pd.DataFrame(results_baseline)
    df_tuned = pd.DataFrame(results_model_a)
    
    # 汇总为论文要求的对比表格数据
    report = pd.DataFrame({
        "模型版本": ["Baseline (原生模型)", "Model A (单次微调)"],
        "情感分类准确率": [f"{df_base['Emotion_Acc'].mean()*100:.2f}%", f"{df_tuned['Emotion_Acc'].mean()*100:.2f}%"],
        "意图识别准确率": [f"{df_base['Intent_Acc'].mean()*100:.2f}%", f"{df_tuned['Intent_Acc'].mean()*100:.2f}%"],
        "情感强度 MAE": [mean_absolute_error(df_base["GT_E_Score"], df_base["Pred_E_Score"]), 
                         mean_absolute_error(df_tuned["GT_E_Score"], df_tuned["Pred_E_Score"])],
        "意图紧急度 MAE": [mean_absolute_error(df_base["GT_I_Score"], df_base["Pred_I_Score"]), 
                          mean_absolute_error(df_tuned["GT_I_Score"], df_tuned["Pred_I_Score"])]
    })

    print("\n\n" + "═"*65)
    print("🏆 【消融实验】核心指标横向对比战报 🏆")
    print("═"*65)
    # 打印格式化的表格，你可以直接截图贴进论文！
    print(report.to_markdown(index=False))
    print("═"*65)
    
    # 保存原始数据，以防你需要画散点图或折线图
    df_all = pd.concat([df_base, df_tuned])
    df_all.to_excel(OUTPUT_EXCEL, index=False)
    print(f"💡 原始测试明细已保存至: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    run_ablation_study()