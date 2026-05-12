# -*- coding: utf-8 -*-
import streamlit as st
import time
import sys
import os

# 路径配置：确保能找到 core 和 utils 文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

from core.engine import NegotiationEngine
from utils.helpers import save_analysis_result

# ==========================================
# 🎨 1. 页面高级配置与全局 CSS 美化
# ==========================================
st.set_page_config(page_title="灵犀 | 意图情绪智能建模引擎", layout="wide", page_icon="🌌")

st.markdown("""
    <style>
    .stApp { background-color: #f7f9fc; font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif; }
    header {visibility: hidden;}
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #eef2f6; box-shadow: 2px 0 10px rgba(0,0,0,0.02); }
    .report-card { background: linear-gradient(145deg, #ffffff, #f0f7ff); border-left: 5px solid #2b6cb0; padding: 20px; border-radius: 10px; font-size: 0.95rem; color: #2d3748; box-shadow: 0 4px 6px rgba(0,0,0,0.05); line-height: 1.6; margin-bottom: 10px; }
    .warm-response { background: linear-gradient(145deg, #ffffff, #fff5f5); border-left: 5px solid #fc8181; padding: 25px; border-radius: 12px; color: #2d3748; box-shadow: 0 4px 15px rgba(252, 129, 129, 0.1); font-size: 1.05rem; line-height: 1.8; margin-top: 15px; }
    .stage-badge { display: inline-block; padding: 4px 10px; border-radius: 15px; font-size: 0.8rem; font-weight: 600; margin-bottom: 10px; }
    .badge-r1 { background-color: #e2e8f0; color: #2b6cb0; }
    .badge-r2 { background-color: #feebc8; color: #c53030; }
    .badge-r3 { background-color: #e1f7e7; color: #166534; }
    details summary { outline: none; font-weight: bold; color: #4A90E2; cursor: pointer; }
    .hero-title { background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.2rem; font-weight: 800; margin-bottom: 10px;}
    .feature-box { background: white; padding: 20px; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px rgba(0,0,0,0.02); }
    
    /* 侧边栏专属 CSS */
    .agent-status-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .agent-item { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; font-size: 0.85rem; color: #4A5568; }
    .agent-name { display: flex; align-items: center; font-weight: 500; }
    .agent-icon { margin-right: 8px; font-size: 1.1rem; }
    .status-online { height: 8px; width: 8px; background-color: #48BB78; border-radius: 50%; display: inline-block; box-shadow: 0 0 5px #48BB78; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# ⚙️ 2. 初始化核心引擎
# ==========================================
if "engine" not in st.session_state:
    st.session_state.engine = NegotiationEngine()
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 🧭 3. 侧边栏：工业级系统面板
# ==========================================
# ==========================================
# 🧭 3. 侧边栏：工业级系统面板
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align:center; color:#2D3748;'>🧠 灵犀 (LingXi)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#718096; font-size:0.85rem; margin-top:-15px;'>基于 LLM 的意图情绪建模引擎 v2.0</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .agent-status-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .agent-item { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; font-size: 0.85rem; color: #4A5568; }
    .agent-name { display: flex; align-items: center; font-weight: 500; white-space: nowrap; }
    .agent-icon { margin-right: 8px; font-size: 1.1rem; }
    .status-online { height: 8px; width: 8px; background-color: #48BB78; border-radius: 50%; display: inline-block; box-shadow: 0 0 5px #48BB78; flex-shrink: 0; }
    </style>
    """, unsafe_allow_html=True)

    current_rounds = len(st.session_state.messages) // 2 
    
# 注意：下面这里的 HTML 必须全部顶格写，绝对不能有空格缩进！
    agent_html = f"""<div class="agent-status-card">
<div style="font-weight: bold; color: #2D3748; margin-bottom: 15px; font-size: 0.95rem; border-bottom: 1px solid #E2E8F0; padding-bottom: 10px;">
🤖 多智能体集群状态
</div>
<div class="agent-item">
<div class="agent-name"><span class="agent-icon">📍</span>Agent 1: 感知层</div>
<span class="status-online"></span>
</div>
<div class="agent-item">
<div class="agent-name"><span class="agent-icon">🎯</span>Agent 2: 决策层</div>
<span class="status-online"></span>
</div>
<div class="agent-item">
<div class="agent-name"><span class="agent-icon">🔍</span>Agent 3: 审计层</div>
<span class="status-online"></span>
</div>
<div class="agent-item">
<div class="agent-name"><span class="agent-icon">📊</span>Agent 4: 重构层</div>
<span class="status-online"></span>
</div>
<div class="agent-item">
<div class="agent-name"><span class="agent-icon">💌</span>Agent 5: 响应层</div>
<span class="status-online"></span>
</div>
<div style="border-top: 1px dashed #CBD5E0; margin: 12px 0;"></div>
<div class="agent-item" style="color: #3182CE; font-weight: 600;">
<div class="agent-name">🔗 上下文轮数</div>
<div>{current_rounds} 轮</div>
</div>
</div>"""

    # 渲染去除了缩进的 HTML
    st.markdown(agent_html, unsafe_allow_html=True)
    
    if st.button("🧹 开启全新评估会话", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 💬 4. 历史记录渲染
# ==========================================
if len(st.session_state.messages) == 0:
    st.markdown("<div style='margin-top: 5vh;'></div>", unsafe_allow_html=True)
    st.markdown("<h1 class='hero-title'>洞察言辞背后的因果逻辑</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #718096; font-size: 1.1rem; margin-bottom: 40px;'>本系统首创“由情绪驱动意图”的序列因果建模架构，为您提供极具深度的心理关联解析。</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='feature-box'><h3 style='color:#4299E1;font-size:1.2rem;'>🎯 意图精准识别</h3><p style='color:#718096; font-size:0.9rem;'>剥离表面语义，直击底层诉求。</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='feature-box'><h3 style='color:#9F7AEA;font-size:1.2rem;'>🌪️ 情绪维度量化</h3><p style='color:#718096; font-size:0.9rem;'>1-10分细粒度情感强度评估。</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='feature-box'><h3 style='color:#F56565;font-size:1.2rem;'>⚖️ 多智能体审计</h3><p style='color:#718096; font-size:0.9rem;'>首创双Agent架构，互相校验逻辑。</p></div>", unsafe_allow_html=True)
else:
    st.title("🧠 “灵犀”意图情绪关联助手")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                if "r_all" in msg:
                    st.markdown(msg["r_all"], unsafe_allow_html=True)
                st.markdown(f"""
                <div class="warm-response">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight:bold; color: #E53E3E; font-size: 1.1rem;">💌 灵犀寄语</span>
                    </div>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

# ==========================================
# 🚀 5. 核心交互流：全生命周期逐字打字机
# ==========================================
if prompt := st.chat_input("说点什么... (例如: 手机丢了，没办法支付了)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        txt_r1, txt_r2, txt_r3, txt_r4, txt_r5 = "", "", "", "", ""
        
        status_box = st.status("🛸 灵犀正在调用多智能体神经中枢...", expanded=True)
        with status_box:
            col1, col2, col3 = st.columns(3)
            p1 = col1.empty()
            p2 = col2.empty()
            p3 = col3.empty()
            
        report_expander = st.empty()
        response_area = st.empty()

        # 对接核心引擎流式获取结果
        for stage, token in st.session_state.engine.run_full_pipeline(prompt, st.session_state.messages[:-1]):
            # Agent 1: 情感感知
            if stage == "R1_TOKEN":
                for char in token:
                    txt_r1 += char
                    p1.markdown(f"<div class='stage-badge badge-r1'>📍感知层：量化</div><br><span style='font-size:0.85rem;'>{txt_r1}▌</span>", unsafe_allow_html=True)
                    time.sleep(0.01)
            
            # Agent 2: 意图推导
            elif stage == "R2_TOKEN":
                for char in token:
                    txt_r2 += char
                    p2.markdown(f"<div class='stage-badge badge-r2'>🎯决策层：推导</div><br><span style='font-size:0.85rem;'>{txt_r2}▌</span>", unsafe_allow_html=True)
                    time.sleep(0.01)
            
            # Agent 3: 逻辑审计
            elif stage == "R3_TOKEN":
                for char in token:
                    txt_r3 += char
                    p3.markdown(f"<div class='stage-badge badge-r3'>🔍审计层：校验</div><br><span style='font-size:0.85rem;'>{txt_r3}▌</span>", unsafe_allow_html=True)
                    time.sleep(0.01)
            
            # Agent 4: 最终 JSON 报告
            elif stage == "R4_TOKEN":
                for char in token:
                    txt_r4 += char
                    report_expander.markdown(f'''
                    <details open style="margin-top: 10px; border: 1px solid #E2E8F0; border-radius: 8px; padding: 15px; background: white;">
                        <summary>📊 查看【意图-情绪】深度关联建模报告</summary>
                        <div class="report-card" style="margin-top: 15px;">{txt_r4}▌</div>
                    </details>
                    ''', unsafe_allow_html=True)
                    time.sleep(0.005) 
            
            # Agent 5: 灵犀寄语
            elif stage == "R5_TOKEN":
                for char in token:
                    txt_r5 += char
                    response_area.markdown(f"""
                    <div class="warm-response">
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <span style="font-weight:bold; color: #E53E3E; font-size: 1.1rem;">💌 灵犀寄语</span>
                        </div>
                        {txt_r5}▌
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.01)

        # ==========================================
        # ✅ 收尾：移除所有光标并固化状态
        # ==========================================
        status_box.update(label="✨ 多级推理与情感建模完成", state="complete", expanded=False)
        
        p1.markdown(f"<div class='stage-badge badge-r1'>📍感知层：量化</div><br><span style='font-size:0.85rem;'>{txt_r1}</span>", unsafe_allow_html=True)
        p2.markdown(f"<div class='stage-badge badge-r2'>🎯决策层：推导</div><br><span style='font-size:0.85rem;'>{txt_r2}</span>", unsafe_allow_html=True)
        p3.markdown(f"<div class='stage-badge badge-r3'>🔍审计层：校验</div><br><span style='font-size:0.85rem;'>{txt_r3}</span>", unsafe_allow_html=True)
        
        report_expander.markdown(f'''
        <details style="margin-top: 10px; border: 1px solid #E2E8F0; border-radius: 8px; padding: 15px; background: white;">
            <summary>📊 查看【意图-情绪】深度关联建模报告</summary>
            <div class="report-card" style="margin-top: 15px;">{txt_r4}</div>
        </details>
        ''', unsafe_allow_html=True)
        
        response_area.markdown(f"""
        <div class="warm-response">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="font-weight:bold; color: #E53E3E; font-size: 1.1rem;">💌 灵犀寄语</span>
            </div>
            {txt_r5}
        </div>
        """, unsafe_allow_html=True)

        # 💾 记忆持久化快照
        process_snapshot = f'''
        <details style="margin-bottom: 10px; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; background: white;">
            <summary>🛸 查看多智能体协同推理快照</summary>
            <div style="display: flex; gap: 10px; margin-top: 15px;">
                <div style="flex: 1;"><div class='stage-badge badge-r1'>📍感知</div><br><span style="font-size:0.8rem;">{txt_r1}</span></div>
                <div style="flex: 1;"><div class='stage-badge badge-r2'>🎯推导</div><br><span style="font-size:0.8rem;">{txt_r2}</span></div>
                <div style="flex: 1;"><div class='stage-badge badge-r3'>🔍审计</div><br><span style="font-size:0.8rem;">{txt_r3}</span></div>
            </div>
        </details>
        <details style="margin-bottom: 10px; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; background: white;">
            <summary>📊 查看【意图-情绪】底层因果建模数据</summary>
            <div class="report-card" style="margin-top: 15px;">{txt_r4}</div>
        </details>
        '''
        
        st.session_state.messages[-1] = {
            "role": "assistant", 
            "content": txt_r5,
            "r_all": process_snapshot
        }
        
        # 保存日志
        save_analysis_result(prompt, txt_r4)
        
        st.rerun()