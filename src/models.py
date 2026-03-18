from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class EmailItem:
    uid: int
    date: datetime
    from_addr: str
    subject: str
    text: str
    mailbox: str
    message_id: Optional[str] = None
    in_reply_to: Optional[str] = None       # 直接父邮件 message_id
    references: Optional[str] = None        # 完整父链 message_id 列表（空格分隔）
    to_addrs: List[str] = field(default_factory=list)
    cc_addrs: List[str] = field(default_factory=list)
    historical_context: str = ""            # RAG 检索后注入的历史上下文


@dataclass
class EmailThread:
    thread_id: str
    subject_fingerprint: str
    participants: List[str]
    items: List[EmailItem]


@dataclass
class ClusterResult:
    threads: List[EmailThread]

