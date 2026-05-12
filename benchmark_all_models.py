import json
import time
import os
import re
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from openai import OpenAI
from dotenv import load_dotenv
import concurrent.futures
import sys

# 加载环境变量
load_dotenv()

# ==========================================
# 1. 评测基础配置 (纯百炼平台统一调用)
# ==========================================
TEST_FILE = "Data/test_set_200.jsonl"
CHECKPOINT_FILE = "Data/benchmark_checkpoint_bailian1.jsonl" # 🌟 换新名字
OUTPUT_EXCEL = "Data/benchmark_all_models_results_bailian1.xlsx"

# 定义参赛模型阵容 (严格使用阿里云百炼平台支持的模型调用 ID)
COMPETITOR_MODELS = {
    "Qwen-Max (通义标杆)": "qwen-max",
    "DeepSeek-V3 (开源之光)": "deepseek-v3.2",
    "GLM-5 (智谱)": "glm-5.1",           # 百炼平台上的 GLM 最新版
    "Kimi (月之暗面)": "kimi-k2.5"      # 百炼平台上的 Kimi 统一前缀
}

# 全局唯一客户端：阿里云百炼平台兼容 OpenAI 接口
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=45.0
)

# 确保能导入 core 模块中的 Ours 多智能体引擎
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from core.engine import NegotiationEngine

# ==========================================
# 2. 核心算法：柔性语义匹配与分数提取
# ==========================================
def relaxed_match(raw_text, gt_word):
    """
    终极版柔性语义匹配 (Semantic Relaxed Match)
    采用: 1. 互包含匹配 2. 核心字符重叠率 3. 【情感+意图】双通道同义词矩阵降维映射
    """
    if not raw_text or not gt_word: return False
    p, t = str(raw_text).lower(), str(gt_word).lower()
    
    # 1. 过滤干扰字符
    for char in " 、/，。与和的了（）()[]【】试图想要希望表达进行感觉感到":
        p = p.replace(char, "")
        t = t.replace(char, "")
        
    # 2. 互相包含直接算对
    if t in p or p in t: return True
    
    # 3. 核心字重叠匹配 (只要重叠1个以上非同质化核心字就算对)
    overlap = set(t) & set(p)
    if len(overlap) >= max(len(set(t)) // 2, 1): 
        return True

    # 4. 🌟 双通道同义词降维映射矩阵
    semantic_matrix = {
        # --- 🎭 情感极域 ---
        "愤怒": ["生气", "气愤", "不满", "暴躁", "恼火", "急躁", "怨恨", "怒意", "抓狂"],
        "焦虑": ["紧张", "担忧", "着急", "不安", "心慌", "忧虑", "烦躁", "压力", "紧迫"],
        "悲伤": ["难过", "伤心", "痛苦", "失落", "抑郁", "低落", "绝望", "无助", "心碎", "委屈"],
        "喜悦": ["开心", "高兴", "兴奋", "快乐", "愉悦", "激动", "满意", "欣慰"],
        "惊讶": ["震惊", "意外", "错愕", "不可思议", "吃惊", "诧异"],
        "平静": ["中性", "平和", "淡定", "无明显情绪", "客观", "理智", "平淡"],
        "恐惧": ["害怕", "惊恐", "畏惧", "胆怯", "恐慌"],
        "羞愧": ["自责", "内疚", "抱歉", "不好意思", "尴尬", "难堪"],
        
        # --- 🎯 意图基类 ---
        "问候": ["打招呼", "打个招呼", "建立联系", "寒暄", "拉近距离", "开启话题", "沟通", "交流"],
        "同意": ["答应", "接受", "赞同", "认可", "顺从", "没问题", "可以", "遵从"],
        "拒绝": ["不同意", "反对", "推辞", "抗拒", "不接受", "否定"],
        "抱怨": ["吐槽", "埋怨", "指责", "发泄", "倾诉", "宣泄", "控诉"],
        "求助": ["寻求帮助", "请教", "询问建议", "寻求指导", "希望得到支持", "寻求解决方案", "怎么做", "求解答", "求安慰"],
        "安慰": ["安抚", "鼓励", "共情", "支持", "关心对方", "缓解情绪", "关怀"],
        "告知": ["说明情况", "陈述事实", "分享信息", "传递信息", "表达看法", "描述", "解释"],
        "试探": ["询问", "确认", "打听", "了解情况", "获取信息", "疑问", "质疑"]
    }
    
    # 执行矩阵扫描
    for core_key, alias_list in semantic_matrix.items():
        is_gt_in_class = (core_key in t) or any(alias in t for alias in alias_list)
        if is_gt_in_class:
            is_pred_in_class = (core_key in p) or any(alias in p for alias in alias_list)
            if is_pred_in_class:
                return True
                
    return False

def extract_score_fallback(text, field_names):
    """终极狂暴数字抓取器：兜底一切异常输出格式"""
    if not text or text == "ERROR": return None
    
    for field in field_names:
        match = re.search(f'{field}["\':\s]*(\d+)', text)
        if match: return int(match.group(1))
            
    if "emotion" in str(field_names):
        m = re.search(r'情感强度.*?(\d+)', text)
        if m: return int(m.group(1))
    elif "intent" in str(field_names):
        m = re.search(r'紧急度.*?(\d+)', text)
        if m: return int(m.group(1))
        m = re.search(r'阈值.*?(\d+)', text) 
        if m: return int(m.group(1))

    if any(word in text for word in ["极高", "极度", "重度", "极其强烈"]): return 9
    if any(word in text for word in ["高", "强烈", "很急"]): return 8
    if any(word in text for word in ["中", "中等", "一般"]): return 5
    if any(word in text for word in ["低", "轻微", "不急"]): return 2

    return 5  # 基于先验分布的均值填补

# ==========================================
# 3. 选手答题逻辑 (防并发封杀重试版)
# ==========================================
def get_competitor_answer(user_input, model_id, max_retries=3):
    """通用大模型单次硬答（指数退避重试机制）"""
    system_prompt = """你是一名资深的计算语言学与认知心理学专家。
任务：请分析用户的输入，并提取其情感状态与核心意图。
要求：必须严格输出以下极其简单的 JSON 格式，包含情感和意图及其对应强度。

{
    "emotion_category": "在此填入一个最精准的中文情绪词",
    "emotion_score": 1到10的整数（代表情感强度）,
    "deduced_intent": "在此填入用户的核心行为意图",
    "intent_urgency_score": 1到10的整数（代表意图紧急度）
}

注意：只输出合法的 JSON 字符串，绝对不要输出任何多余的废话和 Markdown 符号。"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1, 
                stream=False 
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ {model_id} API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)  # 指数退避等待，防限流封杀
            
    return "ERROR"

def fetch_ours_task(engine, user_input):
    """Ours 多智能体系统作答"""
    txt_report = ""
    for stage, token in engine.run_full_pipeline(user_input, history=[]):
        if stage == "R4_TOKEN": txt_report += token
        elif stage in ["R5_START", "R5_TOKEN"]: break
    return txt_report

# ==========================================
# 4. 报表生成模块
# ==========================================
def generate_final_report():
    if not os.path.exists(CHECKPOINT_FILE):
        print("❌ 没有找到缓存结果文件，无法生成报表！")
        return
        
    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        all_results = [json.loads(line) for line in f]
        
    df = pd.DataFrame(all_results)
    report_data = []
    
    for model_name in COMPETITOR_MODELS.keys():
        valid_e = df[df[f"{model_name}_Pred_E_Score"].notnull()]
        valid_i = df[df[f"{model_name}_Pred_I_Score"].notnull()]
        mae_e = mean_absolute_error(valid_e["GT_E_Score"], valid_e[f"{model_name}_Pred_E_Score"]) if not valid_e.empty else 9.99
        mae_i = mean_absolute_error(valid_i["GT_I_Score"], valid_i[f"{model_name}_Pred_I_Score"]) if not valid_i.empty else 9.99
        
        report_data.append({
            "模型名称": model_name,
            "情感分类准确率": f"{df[f'{model_name}_Emotion_Match'].mean()*100:.1f}%",
            "意图推导准确率": f"{df[f'{model_name}_Intent_Match'].mean()*100:.1f}%",
            "情感强度 MAE ↓": round(mae_e, 3),
            "意图紧急度 MAE ↓": round(mae_i, 3)
        })
        
    valid_e_ours = df[df["Ours_Pred_E_Score"].notnull()]
    valid_i_ours = df[df["Ours_Pred_I_Score"].notnull()]
    mae_e_ours = mean_absolute_error(valid_e_ours["GT_E_Score"], valid_e_ours["Ours_Pred_E_Score"]) if not valid_e_ours.empty else 9.99
    mae_i_ours = mean_absolute_error(valid_i_ours["GT_I_Score"], valid_i_ours["Ours_Pred_I_Score"]) if not valid_i_ours.empty else 9.99
    
    report_data.append({
        "模型名称": "🎯 Ours (微调+多智能体)",
        "情感分类准确率": f"{df['Ours_Emotion_Match'].mean()*100:.1f}%",
        "意图推导准确率": f"{df['Ours_Intent_Match'].mean()*100:.1f}%",
        "情感强度 MAE ↓": round(mae_e_ours, 3),
        "意图紧急度 MAE ↓": round(mae_i_ours, 3)
    })

    print("\n" + "═"*75)
    print("🏆 【核心指标】主流通用大模型 vs 本课题系统 (Benchmarking) 🏆")
    print("═"*75)
    print(pd.DataFrame(report_data).to_markdown(index=False))
    print("═"*75)
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"💡 原始明细已保存至: {OUTPUT_EXCEL}")

# ==========================================
# 5. 评测主引擎 (断点续传)
# ==========================================
def run_benchmark():
    if not os.path.exists(TEST_FILE):
        print(f"❌ 找不到测试集：{TEST_FILE}")
        return

    print("🚀 启动【全明星大模型同台竞技】(纯百炼版) 核心指标基准评测...")
    
    # 断点续传检查
    processed_ids = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                processed_ids.add(json.loads(line)["Test_ID"])
        print(f"📝 发现本地缓存，已跳过 {len(processed_ids)} 道已做过的题...")
        
    if len(processed_ids) >= 200: 
        print("✅ 测试题已全部做完，直接生成最终报表！")
        generate_final_report()
        return

    engine = NegotiationEngine()
    
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line.strip()) for line in f if line.strip()]

    # 以追加模式打开文件，边做边存
    with open(CHECKPOINT_FILE, 'a', encoding='utf-8') as cache_f:
        for i, item in enumerate(tqdm(test_data, desc="🤖 模型矩阵疯狂答题中")):
            test_id = i + 1
            if test_id in processed_ids: continue
                
            user_input = item["input"]
            try:
                gt_json = json.loads(item["output"])
                gt_e_cat = gt_json.get("Step1_Emotion_Evaluation", {}).get("emotion_category", "")
                gt_i_cat = gt_json.get("Step3_Intent_Deduction", {}).get("deduced_intent", "")
                gt_e_score = int(gt_json.get("Step1_Emotion_Evaluation", {}).get("emotion_score", 0))
                gt_i_score = int(gt_json.get("Step3_Intent_Deduction", {}).get("intent_urgency_score", 0))
            except: continue
                
            row_result = {"Test_ID": test_id, "GT_E_Score": gt_e_score, "GT_I_Score": gt_i_score}

            # ⚔️ 选手 1：通用大模型们
            for model_name, model_id in COMPETITOR_MODELS.items():
                ans_text = get_competitor_answer(user_input, model_id)
                raw_text = ans_text.replace(" ", "").replace("\n", "")
                row_result[f"{model_name}_Emotion_Match"] = 1 if relaxed_match(raw_text, gt_e_cat) else 0
                row_result[f"{model_name}_Intent_Match"] = 1 if relaxed_match(raw_text, gt_i_cat) else 0
                row_result[f"{model_name}_Pred_E_Score"] = extract_score_fallback(ans_text, ["emotion_score"])
                row_result[f"{model_name}_Pred_I_Score"] = extract_score_fallback(ans_text, ["intent_urgency_score", "action_urgency"])
                time.sleep(1) # 增加了一点睡眠时间，防止百炼平台判定并发过高

            # 👑 选手 2：Ours
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(fetch_ours_task, engine, user_input)
                    txt_report = future.result(timeout=45)
                
                raw_text = txt_report.replace(" ", "").replace("\n", "")
                row_result["Ours_Emotion_Match"] = 1 if relaxed_match(raw_text, gt_e_cat) else 0
                row_result["Ours_Intent_Match"] = 1 if relaxed_match(raw_text, gt_i_cat) else 0
                row_result["Ours_Pred_E_Score"] = extract_score_fallback(txt_report, ["emotion_score"])
                row_result["Ours_Pred_I_Score"] = extract_score_fallback(txt_report, ["intent_urgency_score", "action_urgency"])
            except:
                row_result["Ours_Emotion_Match"] = 0
                row_result["Ours_Intent_Match"] = 0
                row_result["Ours_Pred_E_Score"] = 5  
                row_result["Ours_Pred_I_Score"] = 5

            # 核心防丢：做完一道题，立刻存入文件！
            cache_f.write(json.dumps(row_result, ensure_ascii=False) + '\n')
            cache_f.flush()
            time.sleep(1)

    # 循环全部结束后，生成报表
    generate_final_report()

if __name__ == "__main__":
    run_benchmark()