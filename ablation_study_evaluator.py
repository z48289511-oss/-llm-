import json
import time
import os
import re
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, f1_score
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. 评测配置与环境变量读取
# ==========================================
TEST_FILE = "Data/test_set_200.jsonl"
OUTPUT_EXCEL = "Data/ablation_study_results.xlsx"

BASELINE_MODEL = "qwen2.5-7b-instruct"
FINE_TUNED_MODEL_ID = os.getenv("FINE_TUNED_MODEL_ID")
CRITIC_MODEL_ID = os.getenv("BASE_MODEL_ID")

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=60.0
)

# ==========================================
# 2. 柔性语义匹配与特征提取工具
# ==========================================
def relaxed_match(raw_text, gt_word):
    if not raw_text or not gt_word: return False
    p, t = str(raw_text).lower(), str(gt_word).lower()
    for char in " 、/，。与和的了（）()[]【】":
        p = p.replace(char, "")
        t = t.replace(char, "")
    if t in p: return True
    overlap = set(t) & set(p)
    if len(overlap) >= min(len(set(t)), 2): return True
    return False

def extract_score(text, field_name):
    match = re.search(f'"{field_name}"\s*:\s*(\d+|"[^"]+")', text)
    if match:
        val = match.group(1).replace('"', '')
        if val.isdigit(): return int(val)
    return 0

# ==========================================
# 3. 大模型调用逻辑
# ==========================================
SINGLE_TURN_PROMPT = """请执行情绪-意图分析。按以下JSON格式输出：
{"Step1_Emotion_Evaluation": {"emotion_category": "词", "emotion_score": "1-10"}, "Step3_Intent_Deduction": {"deduced_intent": "词", "intent_urgency_score": "1-10"}}
输入："""

def run_single_turn_model(user_input, model_id):
    try:
        resp = client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": SINGLE_TURN_PROMPT + user_input}], temperature=0.1)
        return resp.choices[0].message.content
    except Exception as e: return f"ERROR: {e}"

def run_multi_agent_pipeline(user_input, main_model_id):
    try:
        r1 = client.chat.completions.create(model=main_model_id, messages=[{"role": "system", "content": "情感专家。输出emotion_category, emotion_score"}, {"role": "user", "content": user_input}], temperature=0.1).choices[0].message.content
        r2 = client.chat.completions.create(model=main_model_id, messages=[{"role": "system", "content": "推导专家。输出deduced_intent, intent_urgency_score"}, {"role": "user", "content": f"输入: {user_input}\n情感: {r1}"}], temperature=0.1).choices[0].message.content
        r3 = client.chat.completions.create(model=CRITIC_MODEL_ID, messages=[{"role": "system", "content": "审计专家。找逻辑漏洞。"}, {"role": "user", "content": f"{r1}\n{r2}"}], temperature=0.3).choices[0].message.content
        r4 = client.chat.completions.create(model=main_model_id, messages=[{"role": "system", "content": "整合专家。输出终稿五维JSON。"}, {"role": "user", "content": f"感知:{r1}\n推导:{r2}\n审计:{r3}"}], temperature=0.1).choices[0].message.content
        return r4
    except Exception as e: return f"ERROR: {e}"

# ==========================================
# 4. 核心评测与耗时统计
# ==========================================
def evaluate_model(txt_report, gt):
    if "ERROR" in txt_report: return 0, 0, 0, 0, 0
    raw = txt_report.replace(" ", "").replace("\n", "")
    e_match = 1 if relaxed_match(raw, gt["e_cat"]) else 0
    i_match = 1 if relaxed_match(raw, gt["i_cat"]) else 0
    p_e = extract_score(txt_report, "emotion_score")
    p_i = extract_score(txt_report, "intent_urgency_score")
    # 如果没拿到紧急度，尝试备用键名
    if p_i == 0: p_i = extract_score(txt_report, "urgency") 
    fv = 1 if p_e > 0 and p_i > 0 else 0
    return e_match, i_match, p_e, p_i, fv

def run_ablation_study():
    if not os.path.exists(TEST_FILE): return print(f"❌ 找不到: {TEST_FILE}")

    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line.strip()) for line in f if line.strip()]

    all_results = []
    
    for i, item in enumerate(tqdm(test_data, desc="🤖 模型矩阵答题中")):
        user_input = item["input"]
        try:
            gt_json = json.loads(item["output"])
            gt = {
                "e_cat": gt_json.get("Step1_Emotion_Evaluation", {}).get("emotion_category", ""),
                "i_cat": gt_json.get("Step3_Intent_Deduction", {}).get("deduced_intent", ""),
                "e_score": int(gt_json.get("Step1_Emotion_Evaluation", {}).get("emotion_score", 0)),
                "i_score": int(gt_json.get("Step3_Intent_Deduction", {}).get("intent_urgency_score", 0))
            }
        except: continue
        
        # 封装一个带计时的执行器
        def run_with_timing(func, *args):
            t0 = time.time()
            ans = func(*args)
            t1 = time.time()
            em, im, pes, pis, fv = evaluate_model(ans, gt)
            return em, im, pes, pis, fv, round(t1 - t0, 2)

        # 🛡️ 执行四大模型并记录耗时
        em1, im1, pes1, pis1, fv1, time1 = run_with_timing(run_single_turn_model, user_input, BASELINE_MODEL)
        em2, im2, pes2, pis2, fv2, time2 = run_with_timing(run_single_turn_model, user_input, FINE_TUNED_MODEL_ID)
        em3, im3, pes3, pis3, fv3, time3 = run_with_timing(run_multi_agent_pipeline, user_input, BASELINE_MODEL)
        em4, im4, pes4, pis4, fv4, time4 = run_with_timing(run_multi_agent_pipeline, user_input, FINE_TUNED_MODEL_ID)
        
        all_results.append({
            "Test_ID": i+1, "GT_E_Score": gt["e_score"], "GT_I_Score": gt["i_score"],
            "Base_EM": em1, "Base_IM": im1, "Base_PES": pes1, "Base_PIS": pis1, "Base_FV": fv1, "Base_Time": time1,
            "Lora_EM": em2, "Lora_IM": im2, "Lora_PES": pes2, "Lora_PIS": pis2, "Lora_FV": fv2, "Lora_Time": time2,
            "Agent_EM": em3, "Agent_IM": im3, "Agent_PES": pes3, "Agent_PIS": pis3, "Agent_FV": fv3, "Agent_Time": time3,
            "Ours_EM": em4, "Ours_IM": im4, "Ours_PES": pes4, "Ours_PIS": pis4, "Ours_FV": fv4, "Ours_Time": time4
        })

    # ==========================================
    # 5. 生成终极战报
    # ==========================================
    df = pd.DataFrame(all_results)
    report_data = []
    configs = [("M_base", "Base"), ("M_lora", "Lora"), ("M_agent", "Agent"), ("M_ours", "Ours")]
    
    for name, prefix in configs:
        valid_df = df[df[f"{prefix}_FV"] == 1]
        mae_e = mean_absolute_error(valid_df["GT_E_Score"], valid_df[f"{prefix}_PES"]) if not valid_df.empty else 9.99
        mae_i = mean_absolute_error(valid_df["GT_I_Score"], valid_df[f"{prefix}_PIS"]) if not valid_df.empty else 9.99
        avg_time = df[f"{prefix}_Time"].mean()
        
        report_data.append({
            "模型版本": name,
            "FSR(鲁棒性)": f"{df[f'{prefix}_FV'].mean()*100:.1f}%",
            "意图语义F1": f"{df[f'{prefix}_IM'].mean()*100:.1f}%", 
            "E-MAE(情感误差)": round(mae_e, 3),
            "U-MAE(紧急度误差)": round(mae_i, 3),
            "平均耗时(秒)": round(avg_time, 2)  # 🌟 新增的极具学术价值的参数！
        })

    df = pd.DataFrame(report_data)

# 1. 无论如何，先保命存盘！
    df.to_csv("ablation_study_results_backup.csv", index=False, encoding="utf-8-sig")
    print("✅ 数据已安全保存至 ablation_study_results_backup.csv")

# 2. 然后再去搞花里胡哨的打印
    try:
       print(df.to_markdown(index=False))
    except ImportError:
       print("⚠️ 未安装 tabulate 模块，无法打印 Markdown 表格。")
    print(df) # 退化成普通的 pandas 打印
if __name__ == "__main__":
    run_ablation_study()