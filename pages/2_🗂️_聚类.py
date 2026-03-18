"""页面 2 — 🗂️ 邮件聚类可视化"""
import sys
import os
from collections import Counter

import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ui.components import render_thread_card

st.set_page_config(page_title="聚类 — Email Agent", page_icon="🗂️", layout="wide")

st.title("🗂️ 邮件聚类可视化")

# ── 检查数据 ──────────────────────────────────────────────────────────────────

threads = st.session_state.get("threads", [])
email_items = st.session_state.get("email_items", [])

if not threads:
    st.info("暂无聚类结果。请先在 **💬 对话** 页面发送一条指令，触发邮件抓取与聚类。")
    st.stop()

# ── 统计信息 ──────────────────────────────────────────────────────────────────

total_emails = sum(len(t.items) for t in threads)
total_threads = len(threads)
all_participants = []
for t in threads:
    all_participants.extend(t.participants)
top_participants = Counter(all_participants).most_common(3)

col1, col2, col3, col4 = st.columns(4)
col1.metric("邮件总数", total_emails)
col2.metric("线程数", total_threads)
col3.metric("平均线程长度", f"{total_emails / total_threads:.1f}" if total_threads else "—")
col4.metric("活跃参与者", len(set(all_participants)))

st.divider()

# ── 排序控制 ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔧 显示控制")
    sort_by = st.selectbox("排序方式", ["邮件数量（多→少）", "主题名称（A-Z）", "最新邮件时间"])
    show_ctx = st.toggle("显示历史上下文", value=False)
    filter_min = st.slider("最少邮件数", min_value=1, max_value=10, value=1)

# ── 排序线程 ──────────────────────────────────────────────────────────────────

filtered = [t for t in threads if len(t.items) >= filter_min]

if sort_by == "邮件数量（多→少）":
    filtered.sort(key=lambda t: len(t.items), reverse=True)
elif sort_by == "主题名称（A-Z）":
    filtered.sort(key=lambda t: t.subject_fingerprint)
elif sort_by == "最新邮件时间":
    def _latest(t):
        try:
            return max(it.date for it in t.items)
        except Exception:
            return ""
    filtered.sort(key=_latest, reverse=True)

# ── 展示聚类结果 ──────────────────────────────────────────────────────────────

st.markdown(f"共 **{len(filtered)}** 个线程（筛选后）")

for idx, thread in enumerate(filtered):
    render_thread_card(thread, idx)
