"""Pydantic 结构化输出模型 — Phase 2"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class RewriteOutput(BaseModel):
    """rewrite_node 的结构化输出，由单次 LLM 调用同时产出。"""
    rewritten_instruction: str = Field(
        description="改写后的完整自包含指令，补全代词/省略，供 plan_node 使用"
    )
    sub_queries: List[str] = Field(
        default_factory=list,
        description=(
            "拆解出的子查询列表，用于并行 RAG 检索。"
            "单意图时返回 [rewritten_instruction]；"
            "多实体/比较类查询时返回各实体的独立查询。"
            "例：['A项目上周进展', 'B项目上周进展']"
        ),
    )
    needs_rewrite: bool = Field(
        description="是否实际进行了改写（False = 原话透传，True = 有实质改动）"
    )


class TodoItem(BaseModel):
    title: str                              # 待办事项标题
    priority: Literal["P0", "P1", "P2"]    # 优先级
    deadline: Optional[str] = None          # 截止时间（ISO 格式）
    related_uids: List[int] = []            # 关联邮件 UID
    action_required: bool = False           # 是否需要 Agent 采取行动
    suggested_action: Optional[str] = None  # 建议动作描述


class MeetingItem(BaseModel):
    subject: str                            # 会议主题
    time: Optional[str] = None             # 时间（ISO 格式）
    location: Optional[str] = None         # 地点或会议链接
    participants: List[str] = []           # 参与者列表
    related_uid: Optional[int] = None      # 来源邮件 UID


class FollowUpItem(BaseModel):
    description: str                        # 跟进事项描述
    direction: Literal["sent", "received"]  # sent=我方需跟进; received=对方承诺事项
    due_date: Optional[str] = None         # 预计截止时间
    related_uid: Optional[int] = None


class DraftReply(BaseModel):
    to_uid: int                             # 待回复的邮件 UID
    draft: str                              # 草拟的回复正文
    tone: Literal["formal", "casual", "concise"] = "formal"
    pending_confirmation: bool = True       # 始终需要人工确认后才发送


class AgentOutput(BaseModel):
    todos: List[TodoItem]                   # 待办事项
    meetings: List[MeetingItem] = []        # 提取的会议信息
    follow_ups: List[FollowUpItem] = []     # 需跟进事项（我方/对方承诺）
    draft_replies: List[DraftReply] = []    # 待确认的草拟回复
    vip_alerts: List[str] = []             # VIP 发件人重要邮件摘要
    summary: str                            # 整体摘要（≤200字）
    unresolved: List[str] = []             # 待确认事项
