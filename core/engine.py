# core/engine.py
import os
import json
from .llm_client import LLMClient
from .agents import AgentPrompts

class NegotiationEngine:
    def __init__(self):
        self.client = LLMClient()
        self.analyzer_model = os.getenv("FINE_TUNED_MODEL_ID") # 你的专属微调模型 (负责硬核特征抽取)
        self.critic_model = os.getenv("BASE_MODEL_ID")       # 通用大基座模型 (负责逻辑找茬和共情聊天)

    def run_full_pipeline(self, user_input, history=[]):
        # --- 1. 情感量化 (必须是微调模型！) ---
        yield "R1_START", "📍 阶段一：感知层 - 正在量化情绪势能..."
        msgs_e = [{"role": "system", "content": AgentPrompts.EMOTION_SCANNER_SYSTEM}, {"role": "user", "content": user_input}]
        full_e = ""
        # ⚠️ 修正：这里改成了 analyzer_model
        for token in self.client.call_stream(self.analyzer_model, msgs_e):
            full_e += token
            yield "R1_TOKEN", token

        # --- 2. 因果推导 (必须是微调模型！) ---
        yield "R2_START", "🌪️ 阶段二：认知层 - 正在执行阈值比对推导..."
        msgs_i = [{"role": "system", "content": AgentPrompts.INTENT_DEDUCER_SYSTEM},
                  {"role": "user", "content": f"输入: {user_input}\n情感量化结果: {full_e}"}]
        full_i = ""
        # ⚠️ 修正：这里改成了 analyzer_model
        for token in self.client.call_stream(self.analyzer_model, msgs_i):
            full_i += token
            yield "R2_TOKEN", token

        # --- 3. 逻辑审计 (Critic - 这是通用大模型的活儿) ---
        yield "R3_START", "🔍 阶段三：审计层 - 正在进行因果一致性校验..."
        msgs_c = [{"role": "system", "content": AgentPrompts.CRITIC_SYSTEM},
                  {"role": "user", "content": f"初步建模全过程：\n{full_e}\n{full_i}"}]
        full_c = ""
        # ✅ 正确：这里保持 critic_model 不动
        for token in self.client.call_stream(self.critic_model, msgs_c):
            full_c += token
            yield "R3_TOKEN", token

        # --- 4. 终稿整合 (必须是微调模型！它对你的 JSON 格式最敏感) ---
        yield "R4_START", "📊 阶段四：整合层 - 正在输出最终建模报告..."
        msgs_f = [{"role": "system", "content": AgentPrompts.JOINT_MODELING_SYSTEM},
                  {"role": "user", "content": f"输入: {user_input}\n情感: {full_e}\n推导: {full_i}\n审计建议: {full_c}\n请输出最终JSON。"}]
        full_r = ""
        # ✅ 正确：这里保持 analyzer_model 不动
        for token in self.client.call_stream(self.analyzer_model, msgs_f):
            full_r += token
            yield "R4_TOKEN", token

       # ... (前面的 R1 到 R4 保持不变) ...

        # --- 5. 灵犀寄语 (The Responser) ---
        yield "R5_START", "💌 阶段五：响应层 - 正在生成专属共情寄语..."
        
        # ⚠️ 核心修复：把用户的原始输入 (user_input) 明确地传给 Agent 5，强迫它结合具体语境！
        msgs_resp = [{"role": "system", "content": AgentPrompts.RESPONSE_SYSTEM}] + history + [
            {"role": "user", "content": f"这是用户刚才遇到的具体事情: {user_input}\n这是后台生成的深度心理学诊断数据: {full_r}\n请严格结合用户遇到的具体事情（语境），给我一段贴心、具体的建议和安慰。"}
        ]
        
        full_ans = ""
        for token in self.client.call_stream(self.critic_model, msgs_resp):
            full_ans += token
            yield "R5_TOKEN", token