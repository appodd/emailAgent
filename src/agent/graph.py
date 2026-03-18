"""LangGraph StateGraph 主图定义 — Phase 4 + Phase 5"""
from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .state import AgentState
from .nodes import make_nodes
from .tools import build_tools

if TYPE_CHECKING:
    from ..config import Config
    from ..imap_client import IMAPFetcher


def build_graph(fetcher: "IMAPFetcher", config: "Config", store=None):
    """
    构建并编译 LangGraph Agent 图。

    Parameters
    ----------
    fetcher : IMAPFetcher
    config  : Config
    store   : FAISS 向量库（可为 None，RAG 可选）

    Returns
    -------
    CompiledGraph — 可直接 .invoke() 或 .stream()
    """
    tools = build_tools(fetcher, store, config)
    fetch_node, cluster_node, retrieve_node, rewrite_node, plan_node, act_node, summarize_node, reflect_node, router, reflection_router = \
        make_nodes(fetcher, config, tools)

    builder = StateGraph(AgentState)

    # 注册节点
    builder.add_node("fetch",     fetch_node)
    builder.add_node("cluster",   cluster_node)
    builder.add_node("rewrite",   rewrite_node)   # Query Rewrite（意图路由 + 子查询分解）
    builder.add_node("retrieve",  retrieve_node)  # 多路并行 RAG 检索（消费 sub_queries）
    builder.add_node("plan",      plan_node)
    builder.add_node("act",       act_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("reflect",   reflect_node)

    # 固定边
    builder.set_entry_point("fetch")
    builder.add_edge("fetch",    "cluster")
    builder.add_edge("cluster",  "rewrite")   # 聚类后先做 Query Rewrite
    builder.add_edge("rewrite",  "retrieve")  # 改写+分解后执行多路检索
    builder.add_edge("retrieve", "plan")      # 检索结果注入 plan_node
    builder.add_edge("act",      "plan")      # ReAct 循环
    builder.add_edge("summarize", "reflect")  # 输出后自评（F1 可选）

    # 条件边：plan → act or summarize
    builder.add_conditional_edges("plan", router, {
        "act":       "act",
        "summarize": "summarize",
    })

    # 条件边：reflect → plan(修正循环) or END
    builder.add_conditional_edges("reflect", reflection_router, {
        "plan":    "plan",
        "__end__": END,
    })

    # Phase 5：SQLite Checkpointer（对话记忆）
    # SqliteSaver 需要 sqlite3 连接，不能用 from_conn_string（返回上下文管理器）
    try:
        import sqlite3 as _sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        _conn = _sqlite3.connect(config.checkpoint_db_path, check_same_thread=False)
        checkpointer = SqliteSaver(_conn)
    except Exception:
        checkpointer = MemorySaver()

    # write_only tools（写操作）在工具内部要求人工确认，不用 interrupt_before
    return builder.compile(checkpointer=checkpointer)
