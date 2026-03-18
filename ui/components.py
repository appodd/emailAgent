"""可复用 UI 渲染组件 — Streamlit 专用

提供 AgentOutput 各模块的标准化渲染函数。
"""
from __future__ import annotations

from typing import List, Optional

import streamlit as st


# ── 优先级颜色映射 ────────────────────────────────────────────────────────────

_PRIORITY_COLOR = {"P0": "🔴", "P1": "🟠", "P2": "🟢"}
_PRIORITY_BADGE = {
    "P0": ":red[P0]",
    "P1": ":orange[P1]",
    "P2": ":green[P2]",
}


# ── AgentOutput 整体渲染 ──────────────────────────────────────────────────────

def render_agent_output(output) -> None:
    """渲染完整 AgentOutput 对象。"""
    if output is None:
        st.info("本轮无结构化输出")
        return

    if output.summary:
        st.info(f"**摘要**：{output.summary}")

    if output.vip_alerts:
        render_vip_alerts(output.vip_alerts)

    tabs = []
    tab_labels = []
    if output.todos:
        tab_labels.append(f"✅ 待办 ({len(output.todos)})")
    if output.meetings:
        tab_labels.append(f"📅 会议 ({len(output.meetings)})")
    if output.follow_ups:
        tab_labels.append(f"🔔 跟进 ({len(output.follow_ups)})")
    if output.draft_replies:
        tab_labels.append(f"✉️ 草稿 ({len(output.draft_replies)})")
    if output.unresolved:
        tab_labels.append(f"❓ 待确认 ({len(output.unresolved)})")

    if not tab_labels:
        st.caption("无结构化数据")
        return

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    if output.todos:
        with tabs[tab_idx]:
            render_todos(output.todos)
        tab_idx += 1

    if output.meetings:
        with tabs[tab_idx]:
            render_meetings(output.meetings)
        tab_idx += 1

    if output.follow_ups:
        with tabs[tab_idx]:
            render_follow_ups(output.follow_ups)
        tab_idx += 1

    if output.draft_replies:
        with tabs[tab_idx]:
            render_draft_replies(output.draft_replies)
        tab_idx += 1

    if output.unresolved:
        with tabs[tab_idx]:
            render_unresolved(output.unresolved)
        tab_idx += 1


# ── 各子模块渲染 ──────────────────────────────────────────────────────────────

def render_todos(todos) -> None:
    """渲染待办事项列表。"""
    for item in todos:
        badge = _PRIORITY_BADGE.get(item.priority, item.priority)
        deadline_str = f"⏰ {item.deadline}" if item.deadline else ""
        action_str = f"\n  > 💡 {item.suggested_action}" if item.suggested_action else ""

        with st.container(border=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"{badge} &nbsp; **{item.title}**{action_str}")
                if deadline_str:
                    st.caption(deadline_str)
            with col2:
                if item.related_uids:
                    st.caption(f"UID: {', '.join(str(u) for u in item.related_uids)}")


def render_meetings(meetings) -> None:
    """渲染会议信息卡片。"""
    for m in meetings:
        with st.container(border=True):
            st.markdown(f"📅 **{m.subject}**")
            cols = st.columns(3)
            with cols[0]:
                st.caption(f"🕐 {m.time or '时间待定'}")
            with cols[1]:
                st.caption(f"📍 {m.location or '地点待定'}")
            with cols[2]:
                st.caption(f"👥 {', '.join(m.participants) if m.participants else '—'}")


def render_follow_ups(items) -> None:
    """渲染跟进事项（我方发出 / 对方承诺 分组）。"""
    sent = [i for i in items if i.direction == "sent"]
    received = [i for i in items if i.direction == "received"]

    if sent:
        st.markdown("**📤 我方需跟进**")
        for item in sent:
            due = f"  ·  ⏰ {item.due_date}" if item.due_date else ""
            st.markdown(f"- {item.description}{due}")

    if received:
        st.markdown("**📥 对方承诺事项**")
        for item in received:
            due = f"  ·  ⏰ {item.due_date}" if item.due_date else ""
            st.markdown(f"- {item.description}{due}")


def render_draft_replies(drafts) -> None:
    """渲染草拟回复，含确认按钮（disabled，提醒需人工确认后发送）。"""
    for draft in drafts:
        with st.container(border=True):
            st.markdown(f"**回复 UID {draft.to_uid}** &nbsp; `{draft.tone}`")
            st.code(draft.draft, language="")
            st.button(
                "确认发送（功能开发中）",
                key=f"send_{draft.to_uid}",
                disabled=True,
                help="草拟回复仅供参考，实际发送功能暂未开启",
            )


def render_vip_alerts(alerts: List[str]) -> None:
    """渲染 VIP 发件人提醒。"""
    for alert in alerts:
        st.warning(f"⭐ VIP：{alert}")


def render_unresolved(items: List[str]) -> None:
    """渲染待确认事项。"""
    for item in items:
        st.markdown(f"- ❓ {item}")


# ── Query Rewrite 信息展示 ────────────────────────────────────────────────────

def render_rewrite_info(
    original: str,
    rewritten: str,
    sub_queries: List[str],
    retrieved_context: str = "",
) -> None:
    """在 expander 中展示 Query Rewrite 过程及 RAG 召回结果。"""
    with st.expander("🔍 Query Rewrite 详情", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**原始指令**")
            st.code(original, language="")
        with col2:
            st.markdown("**改写结果**")
            st.code(rewritten, language="")

        if sub_queries:
            st.markdown("**子查询列表**")
            for i, q in enumerate(sub_queries, 1):
                st.markdown(f"`{i}.` {q}")

        if retrieved_context:
            st.markdown("**RAG 召回片段**")
            snippets = retrieved_context.split("\n\n---\n\n")
            for snippet in snippets:
                st.markdown(f"> {snippet[:300].replace(chr(10), '  \n> ')}")


# ── 邮件线程卡片 ──────────────────────────────────────────────────────────────

def render_thread_card(thread, idx: int) -> None:
    """渲染单个 EmailThread 折叠卡片。"""
    participants = list(set(thread.participants))[:5]
    participant_str = ", ".join(participants)
    label = f"**{thread.subject_fingerprint}** &nbsp; · &nbsp; {len(thread.items)} 封 &nbsp; · &nbsp; {participant_str}"

    with st.expander(label, expanded=False):
        for item in sorted(thread.items, key=lambda x: x.date):
            date_str = item.date.strftime("%Y-%m-%d %H:%M") if hasattr(item.date, "strftime") else str(item.date)
            st.markdown(f"**{date_str}** &nbsp; `{item.from_addr}`")
            preview = (item.text or "")[:300].replace("\n", " ")
            st.caption(preview + ("..." if len(item.text or "") > 300 else ""))
            if item.historical_context:
                with st.expander("历史上下文", expanded=False):
                    st.caption(item.historical_context[:400])
            st.divider()


# ── 聊天气泡 ──────────────────────────────────────────────────────────────────

def render_chat_message(role: str, content: str, output=None, meta: dict = None) -> None:
    """渲染单条聊天消息，AI 消息附带 AgentOutput 结构化块。"""
    with st.chat_message(role):
        st.markdown(content)
        if output is not None:
            render_agent_output(output)
        if meta:
            render_rewrite_info(
                original=meta.get("instruction", ""),
                rewritten=meta.get("rewritten_instruction", ""),
                sub_queries=meta.get("sub_queries", []),
                retrieved_context=meta.get("retrieved_context", ""),
            )
