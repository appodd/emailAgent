from __future__ import annotations

import re
from email.utils import parseaddr
from typing import List

from bs4 import BeautifulSoup


_RE_PREFIX = re.compile(r"^(re|fw|fwd)\s*[:：\-]\s*", re.IGNORECASE)
_RE_TICKET = re.compile(r"\[(?:[^\]]{1,20})\]|\(#\d+\)|\b[A-Z]{2,5}-\d+\b")
_RE_WS = re.compile(r"\s+")


def normalize_subject(subject: str) -> str:
    if not subject:
        return ""
    s = subject.strip()
    s = _RE_PREFIX.sub("", s)
    s = _RE_TICKET.sub("", s)
    s = _RE_WS.sub(" ", s)
    return s.strip().lower()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_key_sentences(text: str, max_sentences: int = 5) -> List[str]:
    if not text:
        return []
    # 简单启发式：以句号/换行分割，过滤过短行
    parts = re.split(r"[\n\r]+|(?<=[。.!?])\s+", text)
    parts = [p.strip() for p in parts if len(p.strip()) >= 6]
    return parts[:max_sentences]


def email_address_domain(addr: str) -> str:
    name, email = parseaddr(addr or "")
    if "@" in email:
        return email.split("@", 1)[1].lower()
    return ""

