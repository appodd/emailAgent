from __future__ import annotations

import email
from datetime import datetime
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime, getaddresses
from typing import List, Optional

from imapclient import IMAPClient

from .config import Config
from .models import EmailItem
from .state_store import StateStore
from .text_utils import html_to_text


def _decode_mime_header(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _extract_text_from_message(msg: email.message.Message) -> str:
    if msg.is_multipart():
        # 优先纯文本
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                try:
                    return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                except Exception:
                    continue
        # 次选 HTML
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/html":
                try:
                    html = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                    return html_to_text(html)
                except Exception:
                    continue
        # 兜底
        try:
            return msg.get_payload(decode=True).decode("utf-8", errors="replace")
        except Exception:
            return ""
    else:
        ctype = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if not payload:
            return ""
        try:
            text = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
        except Exception:
            text = payload.decode("utf-8", errors="replace")
        if ctype == "text/html":
            return html_to_text(text)
        return text


class IMAPFetcher:
    def __init__(self, config: Config, state: StateStore) -> None:
        self.config = config
        self.state = state

    def _connect(self) -> IMAPClient:
        client = IMAPClient(self.config.imap_host, port=self.config.imap_port, ssl=True, timeout=self.config.request_timeout)
        client.login(self.config.imap_user, self.config.imap_password)
        return client

    def fetch_unread(self, since: datetime, mailbox: Optional[str] = None, ignore_state: bool = False, include_seen: bool = False) -> List[EmailItem]:
        mbox = mailbox or self.config.mailbox
        last_seen_uid = self.state.get_last_seen_uid(mbox)

        with self._connect() as client:
            client.select_folder(mbox, readonly=True)

            criteria = ["SINCE", since.date()]
            if not include_seen:
                criteria.insert(0, "UNSEEN")
            uids = client.search(criteria)

            if not ignore_state:
                uids = [u for u in uids if int(u) > int(last_seen_uid)]
            if not uids:
                return []

            fetch_map = client.fetch(uids, [b"RFC822"])  # 全文

        items: List[EmailItem] = []
        for uid, data in fetch_map.items():
            raw = data.get(b"RFC822")
            if not raw:
                continue
            try:
                msg = email.message_from_bytes(raw)
            except Exception:
                continue

            subject = _decode_mime_header(msg.get("Subject", ""))
            from_addr = _decode_mime_header(msg.get("From", ""))
            to_addrs = [a for n, a in getaddresses(msg.get_all("To", []))]
            cc_addrs = [a for n, a in getaddresses(msg.get_all("Cc", []))]

            # 日期
            dt: Optional[datetime] = None
            try:
                if msg.get("Date"):
                    dt = parsedate_to_datetime(msg.get("Date"))
            except Exception:
                dt = None
            if dt is None:
                dt = since  # 兜底

            text = _extract_text_from_message(msg)

            items.append(
                EmailItem(
                    uid=int(uid),
                    date=dt,
                    from_addr=from_addr,
                    subject=subject,
                    text=text,
                    mailbox=mbox,
                    message_id=msg.get("Message-Id"),
                    to_addrs=to_addrs,
                    cc_addrs=cc_addrs,
                )
            )

        # 更新增量状态（调用方保存）
        return sorted(items, key=lambda x: x.uid)

