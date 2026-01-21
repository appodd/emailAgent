from __future__ import annotations

import json
import os
from typing import Dict, Any, Iterable


class StateStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._state: Dict[str, Any] = {"mailboxes": {}}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self._state = json.load(f)
        self._state.setdefault("mailboxes", {})
        self._loaded = True

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)

    def _mbox(self, mailbox: str) -> Dict[str, Any]:
        self.load()
        mboxes = self._state.setdefault("mailboxes", {})
        box = mboxes.setdefault(mailbox, {})
        box.setdefault("last_seen_uid", 0)
        box.setdefault("processed_uids", [])
        return box

    def get_last_seen_uid(self, mailbox: str) -> int:
        return int(self._mbox(mailbox).get("last_seen_uid", 0))

    def set_last_seen_uid(self, mailbox: str, uid: int) -> None:
        box = self._mbox(mailbox)
        if uid > box.get("last_seen_uid", 0):
            box["last_seen_uid"] = int(uid)

    def is_processed(self, mailbox: str, uid: int) -> bool:
        return int(uid) in set(self._mbox(mailbox).get("processed_uids", []))

    def mark_processed(self, mailbox: str, uids: Iterable[int], max_keep: int = 5000) -> None:
        box = self._mbox(mailbox)
        processed = list(dict.fromkeys(list(box.get("processed_uids", [])) + [int(u) for u in uids]))
        if len(processed) > max_keep:
            processed = processed[-max_keep:]
        box["processed_uids"] = processed

