from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import sys

from dateutil import parser as dtparser

from .config import load_config
from .state_store import StateStore
from .imap_client import IMAPFetcher
from .clusterer import cluster_emails
from .llm_summarizer import summarize_threads
from .renderer import render_document, write_markdown


def _parse_since(arg: str) -> datetime:
    arg = (arg or "").strip().lower()
    now = datetime.now(timezone.utc)
    if arg.endswith("d") and arg[:-1].isdigit():
        return now - timedelta(days=int(arg[:-1]))
    if arg.endswith("h") and arg[:-1].isdigit():
        return now - timedelta(hours=int(arg[:-1]))
    try:
        # 尝试解析绝对时间
        dt = dtparser.parse(arg)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        # 默认 7 天
        return now - timedelta(days=7)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Email Agent: IMAP + 聚类 + LLM 总结/待办")
    p.add_argument("--since", default="7d", help="抓取起始时间，示例: 7d / 48h / 2025-01-01")
    p.add_argument("--mailbox", default=None, help="IMAP 邮箱文件夹，默认使用配置 MAILBOX")
    p.add_argument("--output", default=None, help="输出 Markdown 文件路径，不填仅打印控制台")
    p.add_argument("--instruction", default="根据邮件生成我的待办并按优先级排序", help="给 LLM 的自然语言指令")
    p.add_argument("--rescan", action="store_true", help="忽略增量状态，重新抓取满足条件的邮件")
    p.add_argument("--include-seen", action="store_true", help="包含已读邮件")

    args = p.parse_args(argv)

    config = load_config()
    mailbox = args.mailbox or config.mailbox
    since = _parse_since(args.since)

    state = StateStore(config.state_path)
    fetcher = IMAPFetcher(config, state)

    items = fetcher.fetch_unread(since, mailbox=mailbox, ignore_state=args.rescan, include_seen=args.include_seen)
    if not items:
        doc = render_document("邮件待办", "_没有新的未读邮件。_")
        print(doc)
        if args.output:
            write_markdown(doc, args.output)
        return 0

    # 聚类
    clustered = cluster_emails(items, config)

    # LLM 汇总/待办
    markdown_body = summarize_threads(clustered.threads, args.instruction, config)
    doc = render_document("邮件待办/小结", markdown_body)

    print(doc)
    if args.output:
        write_markdown(doc, args.output)

    # 更新增量状态
    max_uid = max(it.uid for it in items)
    state.set_last_seen_uid(mailbox, max_uid)
    state.mark_processed(mailbox, [it.uid for it in items])
    state.save()

    return 0


if __name__ == "__main__":
    sys.exit(main())


