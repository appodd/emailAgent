"""Agent 执行封装 — Streamlit 专用

封装 LangGraph graph.stream()，处理：
  · graph 懒初始化（单例，避免重复创建）
  · 首轮 / 续轮 state 构造
  · 节点进度回调（供 st.status 显示）
  · 最终 output / state 读取
"""
from __future__ import annotations

import sys
import os
from typing import Callable, Optional

# 确保项目 src 在 path 中
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# 节点名 → 中文进度描述
NODE_LABELS = {
    "fetch":    "📥 抓取邮件...",
    "cluster":  "🗂️ 邮件聚类...",
    "rewrite":  "✏️ Query Rewrite...",
    "retrieve": "🔍 RAG 检索...",
    "plan":     "🤖 规划中...",
    "act":      "🔧 执行工具...",
    "summarize":"📝 生成输出...",
    "reflect":  "🔄 自反思评审...",
}


def build_initial_state(instruction: str) -> dict:
    return {
        "instruction": instruction,
        "rewritten_instruction": "",
        "sub_queries": [],
        "retrieved_context": "",
        "email_items": [],
        "threads": [],
        "output": None,
        "actions_taken": [],
        "messages": [],
        "critique": "",
        "reflection_count": 0,
        "enable_reflection": False,
    }


def build_followup_state(instruction: str) -> dict:
    return {
        "instruction": instruction,
        "rewritten_instruction": "",
        "sub_queries": [],
        "retrieved_context": "",
        "output": None,
        "actions_taken": [],
        "critique": "",
        "reflection_count": 0,
    }


def init_graph(config, fetcher):
    """初始化 LangGraph 图（含向量库加载）。"""
    from src.agent.graph import build_graph
    from src.email_store import load_store

    store = load_store(config)
    graph = build_graph(fetcher, config, store)
    return graph, store


def run_agent(
    graph,
    instruction: str,
    thread_id: str,
    is_first_turn: bool,
    on_node: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    执行一轮 Agent，返回最终 state 字典。

    Parameters
    ----------
    graph         : CompiledGraph
    instruction   : 用户指令
    thread_id     : LangGraph checkpoint session ID
    is_first_turn : 首轮传完整 state，后续轮只传必要字段
    on_node       : 每个节点完成时的回调，接收节点名字符串

    Returns
    -------
    dict — graph.get_state() 的 .values
    """
    graph_config = {"configurable": {"thread_id": thread_id}}

    input_state = (
        build_initial_state(instruction)
        if is_first_turn
        else build_followup_state(instruction)
    )

    for chunk in graph.stream(input_state, config=graph_config, stream_mode="values"):
        # chunk 是当前 state snapshot，通过与上一帧对比判断哪个节点刚完成
        # LangGraph stream values 模式每次节点完成后推送完整 state
        # 我们用 actions_taken 的变化来粗略感知节点执行
        pass

    # 节点执行完成后获取最终状态
    # 使用 stream_mode="updates" 更精确地拿到节点名
    final_values = graph.get_state(graph_config).values
    return final_values


def run_agent_with_progress(
    graph,
    instruction: str,
    thread_id: str,
    is_first_turn: bool,
    status_container=None,
) -> dict:
    """
    执行一轮 Agent，用 Streamlit st.status 显示节点进度。

    Parameters
    ----------
    status_container : st.status 上下文，或 None（不显示进度）
    """
    import streamlit as st

    graph_config = {"configurable": {"thread_id": thread_id}}

    input_state = (
        build_initial_state(instruction)
        if is_first_turn
        else build_followup_state(instruction)
    )

    prev_node_set: set = set()

    for chunk in graph.stream(input_state, config=graph_config, stream_mode="updates"):
        # stream_mode="updates" 下 chunk 是 {node_name: node_output} dict
        for node_name in chunk.keys():
            if node_name not in prev_node_set:
                prev_node_set.add(node_name)
                label = NODE_LABELS.get(node_name, f"⚙️ {node_name}...")
                if status_container is not None:
                    status_container.update(label=label)

    final_values = graph.get_state(graph_config).values
    return final_values
