from __future__ import annotations

import os
from datetime import datetime


def render_document(title: str, body_markdown: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"# {title}\n\n_生成时间：{ts}_\n\n{body_markdown}\n"


def write_markdown(md: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


