"""页面 4 — ⚙️ 配置 & 初始化"""
import sys
import os

import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

st.set_page_config(page_title="设置 — Email Agent", page_icon="⚙️", layout="wide")

st.title("⚙️ 设置 & 初始化")

# ── 从 .env 预填（如果已加载）────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)

# ── 配置表单 ──────────────────────────────────────────────────────────────────

with st.form("config_form"):
    st.subheader("📮 IMAP 连接")
    col1, col2 = st.columns(2)
    with col1:
        imap_host = st.text_input("IMAP_HOST", value=_env("IMAP_HOST", "imap.gmail.com"))
        imap_user = st.text_input("IMAP_USER（邮箱账号）", value=_env("IMAP_USER"))
        mailbox = st.text_input("MAILBOX", value=_env("MAILBOX", "INBOX"))
    with col2:
        imap_port = st.number_input("IMAP_PORT", value=int(_env("IMAP_PORT", "993")), step=1)
        imap_password = st.text_input("IMAP_PASSWORD", value=_env("IMAP_PASSWORD"), type="password")

    st.divider()
    st.subheader("🤖 LLM 配置")
    col3, col4 = st.columns(2)
    with col3:
        deepseek_api_key = st.text_input("DEEPSEEK_API_KEY", value=_env("DEEPSEEK_API_KEY"), type="password")
        deepseek_model = st.text_input("DEEPSEEK_MODEL", value=_env("DEEPSEEK_MODEL", "deepseek-chat"))
    with col4:
        deepseek_base_url = st.text_input("DEEPSEEK_BASE_URL", value=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))

    st.divider()
    st.subheader("🔍 RAG 检索配置")
    col5, col6 = st.columns(2)
    with col5:
        rag_bm25_weight = st.slider("BM25 权重", 0.0, 1.0, float(_env("RAG_BM25_WEIGHT", "0.4")), 0.05)
        rerank_top_n = st.number_input("Rerank Top-N", min_value=1, max_value=10, value=3)
    with col6:
        enable_rerank = st.toggle("启用 Cross-encoder Rerank", value=False,
                                  help="需要本地下载 BAAI/bge-reranker-v2-m3（约 1.1GB）")
        rerank_model = st.text_input(
            "Rerank 模型",
            value="BAAI/bge-reranker-v2-m3",
            disabled=not enable_rerank,
        )

    st.divider()
    st.subheader("🧵 Session 配置")
    thread_id = st.text_input("Session ID", value=st.session_state.get("thread_id", "streamlit-default"),
                              help="同一 Session ID 保留多轮对话记忆")
    since_days = st.slider("抓取最近 N 天邮件", min_value=1, max_value=30, value=7)

    submitted = st.form_submit_button("🚀 初始化 Agent", type="primary", use_container_width=True)

# ── 初始化 Agent ──────────────────────────────────────────────────────────────

if submitted:
    # 基础校验
    missing = []
    if not imap_host: missing.append("IMAP_HOST")
    if not imap_user: missing.append("IMAP_USER")
    if not imap_password: missing.append("IMAP_PASSWORD")
    if not deepseek_api_key: missing.append("DEEPSEEK_API_KEY")

    if missing:
        st.error(f"以下必填项未填写：{', '.join(missing)}")
        st.stop()

    with st.spinner("正在初始化 Agent..."):
        try:
            # 设置环境变量（覆盖 .env）
            os.environ["IMAP_HOST"] = imap_host
            os.environ["IMAP_PORT"] = str(int(imap_port))
            os.environ["IMAP_USER"] = imap_user
            os.environ["IMAP_PASSWORD"] = imap_password
            os.environ["MAILBOX"] = mailbox
            os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
            os.environ["DEEPSEEK_BASE_URL"] = deepseek_base_url
            os.environ["DEEPSEEK_MODEL"] = deepseek_model
            os.environ["RAG_BM25_WEIGHT"] = str(rag_bm25_weight)
            if enable_rerank:
                os.environ["RAG_RERANK_MODEL"] = rerank_model

            from src.config import load_config
            from src.state_store import StateStore
            from src.imap_client import IMAPFetcher
            from ui.agent_runner import init_graph

            config = load_config()
            # 覆盖部分参数
            config.rag_bm25_weight = rag_bm25_weight
            config.rag_rerank_top_n = rerank_top_n
            if not enable_rerank:
                config.rag_rerank_model = None

            state_store = StateStore(config.state_path)
            fetcher = IMAPFetcher(config, state_store)

            graph, store = init_graph(config, fetcher)

            # 写入 session state
            st.session_state.config = config
            st.session_state.fetcher = fetcher
            st.session_state.graph = graph
            st.session_state.store = store
            st.session_state.thread_id = thread_id
            st.session_state.agent_ready = True
            st.session_state.is_first_turn = True
            st.session_state.chat_history = []
            st.session_state.email_items = []
            st.session_state.threads = []
            st.session_state.last_output = None

            st.success(f"✅ Agent 初始化成功！已连接 `{imap_user}@{imap_host}`")
            st.info("前往 **💬 对话** 页面开始使用。")

        except Exception as e:
            st.error(f"初始化失败：{e}")
            import traceback
            st.code(traceback.format_exc())

# ── 当前状态展示 ──────────────────────────────────────────────────────────────

st.divider()
st.subheader("📊 当前状态")

if st.session_state.get("agent_ready"):
    cfg = st.session_state.config
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("IMAP 账号", cfg.imap_user)
    col_b.metric("LLM 模型", cfg.deepseek_model)
    col_c.metric("Session ID", st.session_state.thread_id)

    try:
        from src.email_store import load_store
        store = load_store(cfg)
        if store:
            doc_count = len(list(store.docstore._dict.values()))
            st.metric("向量库邮件数", doc_count)
        else:
            st.info("向量库尚未建立（首次使用时自动创建）")
    except Exception:
        pass
else:
    st.info("尚未初始化，请填写表单后点击「初始化 Agent」")
