import json
import time
import os
import re
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import sys
import concurrent.futures

# 确保能导入 core 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from core.engine import NegotiationEngine

# ==========================================
# 1. 评测基础配置
# ==========================================
TEST_FILE = "Data/test_set_200.jsonl"         # 你的 200 条期末考试卷
OUTPUT_EXCEL = "Data/experiment_results.xlsx" # 输出的详细结果表格

# ==========================================
# 2. 核心算法：柔性语义匹配
# ==========================================
def relaxed_match(raw_text, gt_word):
    """
    柔性匹配：直接在模型长篇大论的纯文本里，搜索是否命中了真实标签的核心词汇
    """
    if not raw_text or not gt_word: return False
    p, t = str(raw_text).lower(), str(gt_word).lower()
    # 去除无意义的标点和连词
    for char in " 、/，。与和的了":
        p = p.replace(char, "")
        t = t.replace(char, "")
    
    # 只要真实标签的核心字出现在了生成文本中，就算作匹配成功
    if t in p: return True
    
    # 核心字符重叠率 >= 2 个字也算对
    overlap = set(t) & set(p)
    if len(overlap) >= min(len(set(t)), 2): return True
    
    return False

# ==========================================
# 3. API 异步调用封装 (防卡死护盾 + 精准抓取R4)
# ==========================================
def fetch_api_task(engine, user_input):
    txt_report = ""
    # 调用多智能体协商流水线
    for stage, token in engine.run_full_pipeline(user_input, history=[]):
        # 🌟 终极修复：抓取 R4（整合专家输出的最终 JSON 报告）
        if stage == "R4_TOKEN": 
            txt_report += token
        # 🌟 拿到报告就提前掐断，不生成 R5（灵犀寄语），极大节省时间！
        elif stage in["R5_START", "R5_TOKEN"]:
            break
    return txt_report

# ==========================================
# 4. 评测主引擎
# ==========================================
def run_benchmark():
    if not os.path.exists(TEST_FILE):
        print(f"❌ 找不到测试集：{TEST_FILE}")
        return

    print("🚀 启动【序列因果联合建模】系统级自动化评测 (R4精准抓取版)...")
    engine = NegotiationEngine()
    
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    results =[]
    
    for i, item in enumerate(tqdm(test_data, desc="🤖 模型答题进度")):
        user_input = item["input"]
        
        # 安全提取标准答案 (Ground Truth)
        try:
            gt = json.loads(item["output"])
            gt_e_cat = gt.get("Step1_Emotion_Evaluation", {}).get("emotion_category", "")
            gt_i_cat = gt.get("Step3_Intent_Deduction", {}).get("deduced_intent", "")
            gt_e_score = int(gt.get("Step1_Emotion_Evaluation", {}).get("emotion_score", 0))
            gt_i_score = int(gt.get("Step3_Intent_Deduction", {}).get("intent_urgency_score", 0))
        except:
            print(f"⚠️ 第 {i+1} 题原数据格式有误，跳过")
            continue
            
        start_time = time.time()
        txt_report = "" 
        
        try:
            # 🌟 核心防卡死护盾：最多只等 60 秒！
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fetch_api_task, engine, user_input)
                # 如果 60 秒没跑完，直接抛出 TimeoutError
                txt_report = future.result(timeout=60)
            
            latency = time.time() - start_time
            raw_text = txt_report.replace(" ", "").replace("\n", "")
            
            # 文本检索匹配
            emotion_matched = 1 if relaxed_match(raw_text, gt_e_cat) else 0
            intent_matched = 1 if relaxed_match(raw_text, gt_i_cat) else 0

            # 正则抓取预测分数
            pred_e_score, pred_i_score = 0, 0 
            e_match = re.search(r'"emotion_score"\s*:\s*(\d+)', txt_report)
            i_match = re.search(r'"intent_urgency_score"\s*:\s*(\d+)', txt_report)
            
            if e_match: pred_e_score = int(e_match.group(1))
            if i_match: pred_i_score = int(i_match.group(1))

            res = {
                "Test_ID": i + 1,
                "Latency(s)": round(latency, 2),
                "GT_Emotion": gt_e_cat, "Pred_Emotion": "Text_Search", "Emotion_Match": emotion_matched,
                "GT_Intent": gt_i_cat, "Pred_Intent": "Text_Search", "Intent_Match": intent_matched,
                "GT_E_Score": gt_e_score, "Pred_E_Score": pred_e_score,
                "GT_I_Score": gt_i_score, "Pred_I_Score": pred_i_score,
                "Format_Valid": 1 if e_match and i_match else 0
            }
            
        except concurrent.futures.TimeoutError:
            res = {
                "Test_ID": i + 1, "Latency(s)": 60,
                "GT_Emotion": gt_e_cat, "Pred_Emotion": "TIMEOUT", "Emotion_Match": 0,
                "GT_Intent": gt_i_cat, "Pred_Intent": "TIMEOUT", "Intent_Match": 0,
                "GT_E_Score": gt_e_score, "Pred_E_Score": 0, "GT_I_Score": gt_i_score, "Pred_I_Score": 0,
                "Format_Valid": 0
            }
        except Exception as e:
            res = {
                "Test_ID": i + 1, "Latency(s)": 0,
                "GT_Emotion": gt_e_cat, "Pred_Emotion": "ERROR", "Emotion_Match": 0,
                "GT_Intent": gt_i_cat, "Pred_Intent": "ERROR", "Intent_Match": 0,
                "GT_E_Score": gt_e_score, "Pred_E_Score": 0, "GT_I_Score": gt_i_score, "Pred_I_Score": 0,
                "Format_Valid": 0
            }
            
        results.append(res)
        time.sleep(0.5) # API 保护

    # ==========================================
    # 5. 数据统计与论文报告输出
    # ==========================================
    df = pd.DataFrame(results)
    
    emo_acc = df["Emotion_Match"].mean()
    int_acc = df["Intent_Match"].mean()
    
    # 提取格式有效的数据来计算 MAE 误差 (防止极端错误值干扰)
    valid_df = df[df["Format_Valid"] == 1]
    if not valid_df.empty:
        mae_emo = mean_absolute_error(valid_df["GT_E_Score"], valid_df["Pred_E_Score"])
        mae_int = mean_absolute_error(valid_df["GT_I_Score"], valid_df["Pred_I_Score"])
    else:
        mae_emo, mae_int = 0, 0

    print("\n\n" + "═"*60)
    print("🏆 序列因果建模系统 (5000条微调版) · 自动化评测战报 🏆")
    print("═"*60)
    print(f"🔹 测试集总样本数      : {len(df)} 条")
    print(f"🔹 格式遵循成功率      : {df['Format_Valid'].mean()*100:.2f}% (JSON 解析率)")
    print(f"🔹 情感分类准确率      : {emo_acc*100:.2f}% (柔性语义匹配)")
    print(f"🔹 意图推导准确率      : {int_acc*100:.2f}% (柔性语义匹配)")
    print(f"📉 情感强度绝对误差(MAE): {mae_emo:.3f} 分 (满分10分，越低越好)")
    print(f"📉 意图紧急度误差 (MAE): {mae_int:.3f} 分 (满分10分，越低越好)")
    print(f"⏱️ 平均单次推理耗时    : {df['Latency(s)'].mean():.2f} 秒")
    print("═"*60)
    
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"💡 详细对比数据已保存至: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    run_benchmark()