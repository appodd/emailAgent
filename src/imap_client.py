from __future__ import annotations

import email
import sys
from datetime import datetime
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime, getaddresses
from typing import List, Optional

# ── Python 3.14 兼容补丁 ──────────────────────────────────────────────────────
# Python 3.14 将 imaplib.IMAP4.file 改为只读属性，imapclient 3.x 仍用赋值语法
# 打补丁使其使用 object.__setattr__ 写入底层 _file，与属性 getter 保持一致
if sys.version_info >= (3, 14):
    import socket as _socket
    import imapclient.tls as _imapclient_tls
    from imapclient.tls import wrap_socket as _wrap_socket

    def _py314_open(self, host: str = "", port: int = 993, timeout=None) -> None:
        self.host = host
        self.port = port
        sock = _socket.create_connection(
            (host, port), timeout if timeout is not None else self._timeout
        )
        self.sock = _wrap_socket(sock, self.ssl_context, host)
        object.__setattr__(self, "_file", self.sock.makefile("rb"))

    def _py314_read(self, size: int) -> bytes:
        return object.__getattribute__(self, "_file").read(size)

    def _py314_readline(self) -> bytes:
        return object.__getattribute__(self, "_file").readline()

    _imapclient_tls.IMAP4_TLS.open     = _py314_open
    _imapclient_tls.IMAP4_TLS.read     = _py314_read
    _imapclient_tls.IMAP4_TLS.readline = _py314_readline
# ──────────────────────────────────────────────────────────────────────────────

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
                    in_reply_to=msg.get("In-Reply-To"),
                    references=msg.get("References"),
                    to_addrs=to_addrs,
                    cc_addrs=cc_addrs,
                )
            )

        # 更新增量状态（调用方保存）
        return sorted(items, key=lambda x: x.uid)

