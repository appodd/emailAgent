"""页面 3 — 🔍 RAG 历史邮件搜索"""
import sys
import os

import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

st.set_page_config(page_title="搜索 — Email Agent", page_icon="🔍", layout="wide")

st.title("🔍 RAG 历史邮件搜索")
st.caption("使用 BM25 + FAISS 混合检索在历史邮件向量库中搜索")

# ── 检查 Agent 初始化 ─────────────────────────────────────────────────────────

if not st.session_state.get("agent_ready"):
    st.warning("⚠️ Agent 尚未初始化，请先前往 **⚙️ 设置** 配置并初始化。")
    st.stop()

config = st.session_state.config

# ── 侧边栏 ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔧 检索配置")
    top_k = st.slider("返回结果数 Top-K", min_value=1, max_value=20, value=5)
    expand_parents = st.toggle("展开父链邮件", value=True,
                               help="检索到回复链中间的邮件时，自动追溯到根邮件")
    st.divider()
    st.caption("BM25 权重由配置文件决定（默认 0.4）")

# ── 搜索框 ────────────────────────────────────────────────────────────────────

query = st.text_input("🔎 输入搜索内容", placeholder="例如：Q4 财务汇报 / 会议时间 / 项目进展")
search_btn = st.button("搜索", type="primary")

if search_btn and query.strip():
    with st.spinner("检索中..."):
        try:
            from src.email_store import load_store, build_hybrid_retriever, expand_with_parents

            store = load_store(config)
            if store is None:
                st.error("向量库尚未建立，请先在 **💬 对话** 页面触发邮件抓取。")
                st.stop()

            all_docs = list(store.docstore._dict.values())
            retriever = build_hybrid_retriever(all_docs, store, config)
            msg_id_index = {
                d.metadata["message_id"]: d
                for d in all_docs
                if d.metadata.get("message_id")
            }

            # 执行检索
            docs = retriever.invoke(query)

            # 可选父链展开
            if expand_parents:
                docs = expand_with_parents(docs, msg_id_index, max_depth=2)

            docs = docs[:top_k]

        except Exception as e:
            st.error(f"检索失败：{e}")
            st.stop()

    # ── 展示结果 ──────────────────────────────────────────────────────────────

    if not docs:
        st.info("未找到相关历史邮件。")
    else:
        st.success(f"找到 {len(docs)} 条相关邮件")
        st.divider()

        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            date_str = meta.get("date", "?")
            from_addr = meta.get("from", "?")
            subject = meta.get("subject", "（无主题）")
            uid = meta.get("uid", "?")

            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{i}. {subject}**")
                    st.caption(f"📅 {date_str} &nbsp;|&nbsp; 📮 {from_addr}")
                with col2:
                    st.caption(f"UID: {uid}")
                    st.caption(f"📬 {meta.get('mailbox', '?')}")

                preview = doc.page_content[:400].replace("\n", " ")
                st.markdown(f"> {preview}{'...' if len(doc.page_content) > 400 else ''}")

                with st.expander("展开全文"):
                    st.text(doc.page_content)

elif search_btn and not query.strip():
    st.warning("请输入搜索内容")

# ── 向量库状态 ────────────────────────────────────────────────────────────────

with st.expander("📊 向量库状态"):
    try:
        from src.email_store import load_store
        store = load_store(config)
        if store:
            doc_count = len(list(store.docstore._dict.values()))
            st.metric("已索引邮件数", doc_count)
            st.caption(f"向量库路径：`{config.email_store_path}`")
            st.caption("Embedding 模型：`sentence-transformers/all-MiniLM-L6-v2`")
        else:
            st.info("向量库尚未建立")
    except Exception as e:
        st.error(f"无法读取向量库：{e}")
