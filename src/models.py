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
    to_addrs: List[str] = field(default_factory=list)
    cc_addrs: List[str] = field(default_factory=list)


@dataclass
class EmailThread:
    thread_id: str
    subject_fingerprint: str
    participants: List[str]
    items: List[EmailItem]


@dataclass
class ClusterResult:
    threads: List[EmailThread]

