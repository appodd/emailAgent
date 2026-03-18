"""页面 1 — 💬 交互式对话（核心页面）"""
import sys
import os

import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ui.components import render_chat_message, render_agent_output, render_rewrite_info

st.set_page_config(page_title="对话 — Email Agent", page_icon="💬", layout="wide")

# ── 检查 Agent 是否已初始化 ───────────────────────────────────────────────────

if not st.session_state.get("agent_ready"):
    st.warning("⚠️ Agent 尚未初始化，请先前往 **⚙️ 设置** 配置并初始化。")
    st.stop()

graph = st.session_state.graph
config = st.session_state.config
thread_id = st.session_state.thread_id

# ── 侧边栏 ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 当前配置")
    st.caption(f"📮 {config.imap_user}")
    st.caption(f"🤖 {config.deepseek_model}")
    st.caption(f"🧵 Session: {thread_id}")
    st.divider()

    enable_reflection = st.toggle("🔄 启用 Self-reflection", value=False)
    show_rewrite = st.toggle("🔍 显示 Query Rewrite 详情", value=True)

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.is_first_turn = True
        st.session_state.last_output = None
        st.session_state.email_items = []
        st.session_state.threads = []
        st.rerun()

    st.divider()
    st.caption("**快捷指令**")
    quick_cmds = [
        "帮我整理今天的邮件并提取待办",
        "有哪些邮件需要我跟进？",
        "提取本周所有会议信息",
        "生成本周邮件周报",
    ]
    for cmd in quick_cmds:
        if st.button(cmd, use_container_width=True, key=f"quick_{cmd}"):
            st.session_state["_quick_input"] = cmd
            st.rerun()

# ── 主区域：对话 ──────────────────────────────────────────────────────────────

st.title("💬 多轮对话")

# 渲染历史对话
for msg in st.session_state.chat_history:
    render_chat_message(
        role=msg["role"],
        content=msg["content"],
        output=msg.get("output"),
        meta=msg.get("meta") if show_rewrite else None,
    )

# ── 接收用户输入 ──────────────────────────────────────────────────────────────

# 处理快捷指令触发的输入
prefill = st.session_state.pop("_quick_input", None)
user_input = st.chat_input("输入指令，例如：帮我整理今天的邮件...")

instruction = prefill or user_input

if instruction:
    # 立即展示用户消息
    with st.chat_message("user"):
        st.markdown(instruction)
    st.session_state.chat_history.append({"role": "user", "content": instruction})

    is_first = st.session_state.is_first_turn

    # 执行 Agent（带进度显示）
    with st.chat_message("assistant"):
        with st.status("🤖 Agent 运行中...", expanded=True) as status:
            from ui.agent_runner import run_agent_with_progress

            final_state = run_agent_with_progress(
                graph=graph,
                instruction=instruction,
                thread_id=thread_id,
                is_first_turn=is_first,
                status_container=status,
            )
            status.update(label="✅ 完成", state="complete", expanded=False)

        output = final_state.get("output")
        rewritten = final_state.get("rewritten_instruction", "")
        sub_queries = final_state.get("sub_queries", [])
        retrieved_context = final_state.get("retrieved_context", "")

        # 更新 session state 中的邮件数据（供聚类页使用）
        if final_state.get("email_items"):
            st.session_state.email_items = final_state["email_items"]
        if final_state.get("threads"):
            st.session_state.threads = final_state["threads"]

        # 显示 Query Rewrite 详情
        if show_rewrite and (rewritten or sub_queries):
            render_rewrite_info(
                original=instruction,
                rewritten=rewritten,
                sub_queries=sub_queries,
                retrieved_context=retrieved_context,
            )

        # 显示 AgentOutput 结构化结果
        if output:
            st.session_state.last_output = output
            render_agent_output(output)
            ai_content = f"已完成分析，共 {len(output.todos)} 个待办、{len(output.meetings)} 个会议。{output.summary}"
        else:
            ai_content = "本轮未生成结构化输出。"
            st.info(ai_content)

    # 保存 AI 消息到历史
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": ai_content,
        "output": output,
        "meta": {
            "instruction": instruction,
            "rewritten_instruction": rewritten,
            "sub_queries": sub_queries,
            "retrieved_context": retrieved_context,
        },
    })

    st.session_state.is_first_turn = False
    st.rerun()
