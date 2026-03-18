"""Streamlit App 入口

运行方式：
    streamlit run app.py
"""
import sys
import os

import streamlit as st

# 确保 src/ 在 path 中
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

st.set_page_config(
    page_title="Email Agent",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State 初始化 ──────────────────────────────────────────────────────

def _init_session():
    defaults = {
        "graph": None,
        "config": None,
        "fetcher": None,
        "store": None,
        "thread_id": "streamlit-default",
        "email_items": [],
        "threads": [],
        "chat_history": [],   # List[dict]: role / content / output / meta
        "last_output": None,
        "is_first_turn": True,
        "agent_ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ── 侧边栏：全局初始化状态 ────────────────────────────────────────────────────

with st.sidebar:
    st.title("📧 Email Agent")
    st.divider()

    if st.session_state.agent_ready:
        cfg = st.session_state.config
        st.success(f"✅ 已连接\n\n`{cfg.imap_user}`@`{cfg.imap_host}`")
        st.caption(f"模型：{cfg.deepseek_model}")
        st.caption(f"Session：{st.session_state.thread_id}")

        if st.button("🔄 重置对话", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.is_first_turn = True
            st.session_state.last_output = None
            st.session_state.email_items = []
            st.session_state.threads = []
            st.rerun()
    else:
        st.warning("⚠️ Agent 尚未初始化\n\n请前往 **⚙️ 设置** 页面配置连接参数后点击「初始化」")

    st.divider()
    st.caption("使用左侧导航栏切换页面")

# ── 主页内容 ──────────────────────────────────────────────────────────────────

st.title("📧 Email Agent")
st.markdown(
    """
欢迎使用 **Email Agent**，一个基于 LangGraph 的智能邮件助理。

### 功能导航

| 页面 | 功能 |
|------|------|
| 💬 对话 | 多轮交互式邮件处理（核心功能） |
| 🗂️ 聚类 | 查看邮件聚类分组结果 |
| 🔍 搜索 | RAG 混合检索历史邮件 |
| ⚙️ 设置 | 配置 IMAP / LLM / RAG 参数 |

### 快速开始

1. 前往 **⚙️ 设置** 填入 IMAP 和 API 配置，点击「初始化 Agent」
2. 前往 **💬 对话** 输入指令，如：`"帮我整理今天的邮件并提取待办"`
3. Agent 自动完成抓取 → 聚类 → 改写 → 检索 → 规划 → 输出

### 架构

```
IMAP 抓取 → cluster → rewrite → retrieve → plan → act → summarize → reflect
```
    """
)
