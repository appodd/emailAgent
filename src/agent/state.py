"""AgentState TypedDict — Phase 4"""
from __future__ import annotations

from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from ..models import EmailItem, EmailThread
from .structured import AgentOutput


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    email_items: List[EmailItem]
    threads: List[EmailThread]
    output: Optional[AgentOutput]
    actions_taken: List[str]
    instruction: str                  # 用户原始指令
    rewritten_instruction: str        # rewrite_node 改写后的指令（无历史时与 instruction 相同）
    sub_queries: List[str]            # 子查询列表（单意图时长度为1），用于并行多路 RAG 检索
    retrieved_context: str            # retrieve_node 多路检索后汇总的文本，注入 plan_node prompt
    # F1 Reflection（按需启用）
    critique: str          # reflect_node 的批评内容，空字符串表示通过
    reflection_count: int  # 已反思次数，上限 2
    enable_reflection: bool  # 是否开启 Reflection
