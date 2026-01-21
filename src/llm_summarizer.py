from __future__ import annotations

from datetime import datetime
from typing import List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config
from .models import EmailThread
 


def _format_thread(t: EmailThread, max_items: int = 6) -> str:
    rows = []
    rows.append(f"主题指纹: {t.subject_fingerprint}")
    if t.participants:
        rows.append(f"参与者: {', '.join(t.participants[:10])}")
    rows.append("时间线与要点:")
    for it in t.items[-max_items:]:
        dt = it.date.isoformat() if isinstance(it.date, datetime) else str(it.date)
        full_text = (it.text or "").strip()
        rows.append(f"- {dt} | From: {it.from_addr} | {full_text}")
    return "\n".join(rows)


def _build_prompt(threads: List[EmailThread], instruction: str) -> list:
    sys = (
        "你是一名资深邮件助理。基于给定邮件线程，总结关键事项，给出明确且可执行的 To-Do 并标注优先级："
        "P0=紧急(≤48h 截止/阻塞他人)、P1=重要但不紧急、P2=一般；"
        "输出为 Markdown 列表，若信息缺失请提出‘待确认’项。"
    )

    threads_block = "\n\n".join(_format_thread(t) for t in threads)

    user = (
        f"指令: {instruction}\n\n"
        f"以下为聚类后的邮件线程：\n\n{threads_block}\n\n"
        "请根据指令输出 Markdown；若可，按 P0→P1→P2 排序。"
    )

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def summarize_threads(threads: List[EmailThread], instruction: str, config: Config) -> str:
    if not threads:
        return "_无可总结的邮件线程。_"

    client = OpenAI(api_key=config.deepseek_api_key, base_url=config.deepseek_base_url)
    messages = _build_prompt(threads, instruction)

    resp = client.chat.completions.create(
        model=config.deepseek_model,
        messages=messages,
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()


