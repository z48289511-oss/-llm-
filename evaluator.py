import json
import time
import os
import re
from sklearn.metrics import accuracy_score, mean_absolute_error
import sys

# 引入核心引擎
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from core.engine import NegotiationEngine

# ==========================================
# 1. 评测配置
# ==========================================
TEST_FILE = "Data/test_set_200.jsonl"
RESULT_FILE = "Data/eval_results.jsonl"

print("🚀 启动【意图-情绪联合建模】多智能体自动化评测...")

if not os.path.exists(TEST_FILE):
    print(f"❌ 找不到测试集文件：{TEST_FILE}")
    sys.exit()

# 初始化系统引擎
engine = NegotiationEngine()

# 存储真实标签 (Ground Truth)
y_true_emotion_cat =[]
y_true_intent_cat = []
y_true_emotion_score =[]
y_true_urgency_score = []

# 存储模型预测结果 (Predictions)
y_pred_emotion_cat =[]
y_pred_intent_cat = []
y_pred_emotion_score = []
y_pred_urgency_score =[]

valid_json_count = 0  # 统计成功输出标准 JSON 的格式遵循率

# ==========================================
# 2. 评测主循环
# ==========================================
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    print(f"\n[评测进度 {i+1}/{len(lines)}] 正在进行多轮协商作答...")
    data = json.loads(line.strip())
    user_input = data["input"]
    
    # 提取标准答案
    try:
        gt = json.loads(data["output"])
        gt_e_cat = gt["Step1_Emotion_Evaluation"]["emotion_category"]
        gt_e_score = int(gt["Step1_Emotion_Evaluation"]["emotion_score"])
        gt_i_cat = gt["Step3_Intent_Deduction"]["deduced_intent"]
        gt_i_score = int(gt["Step3_Intent_Deduction"]["intent_urgency_score"])
    except Exception as e:
        print("标准答案解析失败，跳过该题...")
        continue

    # 让模型进行作答 (捕获 R3 的最终五维报告)
    txt_r3 = ""
    try:
        for stage, token in engine.run_full_pipeline(user_input, history=[]):
            if stage == "R3_TOKEN":
                txt_r3 += token
                
        # 清理模型的 Markdown 输出并解析为 JSON
        clean_json = txt_r3.strip()
        if clean_json.startswith("```json"): clean_json = clean_json[7:-3].strip()
        elif clean_json.startswith("```"): clean_json = clean_json[3:-3].strip()
        
        pred = json.loads(clean_json)
        
        # 提取模型预测的答案
        pred_e_cat = pred["Step1_Emotion_Evaluation"]["emotion_category"]
        pred_e_score = int(pred["Step1_Emotion_Evaluation"]["emotion_score"])
        pred_i_cat = pred["Step3_Intent_Deduction"]["deduced_intent"]
        pred_i_score = int(pred["Step3_Intent_Deduction"]["intent_urgency_score"])
        
        valid_json_count += 1
        print(f"✅ 作答成功:[真实情绪分数:{gt_e_score} -> 预测分数:{pred_e_score}]")
        
    except Exception as e:
        print(f"⚠️ 模型作答格式异常或超时，记为错误: {e}")
        # 如果模型崩了，预测给个空值或极端错误值
        pred_e_cat = "FORMAT_ERROR"
        pred_i_cat = "FORMAT_ERROR"
        pred_e_score = 0  
        pred_i_score = 0

    # 记录到总表
    y_true_emotion_cat.append(gt_e_cat)
    y_true_intent_cat.append(gt_i_cat)
    y_true_emotion_score.append(gt_e_score)
    y_true_urgency_score.append(gt_i_score)
    y_pred_emotion_cat.append(pred_e_cat)
    y_pred_intent_cat.append(pred_i_cat)
    y_pred_emotion_score.append(pred_e_score)
    y_pred_urgency_score.append(pred_i_score)
    
    time.sleep(0.5)

# ==========================================
# 3. 统计并打印最终得分 (毕业论文核心数据)
# ==========================================
# ==========================================
# 3. 统计并打印最终得分 (引入学术级柔性评估)
# ==========================================
print("\n" + "="*50)
print("🏆 联合建模系统自动化评测最终报告 🏆")
print("="*50)

def relaxed_accuracy(y_true, y_pred):
    """
    柔性语义匹配算法 (Relaxed Accuracy)
    只要预测结果和标准答案命中了核心字/词，即判定为正确。
    """
    correct = 0
    for t, p in zip(y_true, y_pred):
        # 统一转小写并去除干扰标点
        t_clean = str(t).replace(" ", "").replace("、", "").replace("/", "").replace("与", "")
        p_clean = str(p).replace(" ", "").replace("、", "").replace("/", "").replace("与", "")
        
        # 1. 互相包含直接算对 (如 "焦虑" 包含在 "极度焦虑" 中)
        if t_clean in p_clean or p_clean in t_clean:
            correct += 1
            continue
            
        # 2. 核心字符重叠率计算 (如 "寻求帮助" 和 "求助")
        overlap = set(t_clean) & set(p_clean)
        # 如果命中 2 个以上的核心同义字，算作正确匹配
        if len(overlap) >= min(len(set(t_clean)), 2): 
            correct += 1
            
    return correct / len(y_true) if len(y_true) > 0 else 0

# 使用柔性匹配计算生成式大模型的准确率
acc_emotion = relaxed_accuracy(y_true_emotion_cat, y_pred_emotion_cat)
acc_intent = relaxed_accuracy(y_true_intent_cat, y_pred_intent_cat)

# 强度的平均绝对误差 (MAE) 保持不变，因为数字是可以精确计算的
from sklearn.metrics import mean_absolute_error
mae_emotion = mean_absolute_error(y_true_emotion_score, y_pred_emotion_score)
mae_intent = mean_absolute_error(y_true_urgency_score, y_pred_urgency_score)

format_rate = valid_json_count / len(lines)

print(f"🔹 1. 情绪分类准确率 (Relaxed Emotion Acc): {acc_emotion * 100:.2f}%")
print(f"🔹 2. 意图识别准确率 (Relaxed Intent Acc):  {acc_intent * 100:.2f}%")
print(f"🔹 3. 情绪分数绝对误差 (Emotion MAE):      {mae_emotion:.2f} 分 (越低越好)")
print(f"🔹 4. 意图紧急度绝对误差 (Urgency MAE):    {mae_intent:.2f} 分 (越低越好)")
print(f"🔹 5. JSON格式遵循率 (Format Compliance): {format_rate * 100:.2f}%")
print("="*50)