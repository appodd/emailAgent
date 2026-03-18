from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
import sys

from dateutil import parser as dtparser

from .config import load_config
from .state_store import StateStore
from .imap_client import IMAPFetcher
from .renderer import render_document, write_markdown


# --mode 对应的预设 instruction
_MODE_INSTRUCTIONS: dict[str, str] = {
    "morning-briefing": (
        "请对今天收到的邮件进行早间速览：按紧急度排序，每封邮件一句话概要，"
        "突出今日必须处理的事项（P0），提取所有会议信息，标出 VIP 发件人邮件。"
    ),
    "follow-up-check": (
        "请检查所有我方发出但尚未获得回复的邮件，生成待跟进列表，"
        "标注发件日期和对方的联系信息，按紧迫程度排序。"
    ),
    "weekly-report": (
        "请基于本周的邮件往来，生成一份周报摘要：包括主要决策、待跟进事项、"
        "已完成的沟通、下周需要关注的事项。"
    ),
}


def _parse_since(arg: str) -> datetime:
    arg = (arg or "").strip().lower()
    now = datetime.now(timezone.utc)
    if arg.endswith("d") and arg[:-1].isdigit():
        return now - timedelta(days=int(arg[:-1]))
    if arg.endswith("h") and arg[:-1].isdigit():
        return now - timedelta(hours=int(arg[:-1]))
    try:
        dt = dtparser.parse(arg)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return now - timedelta(days=7)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Email Agent: IMAP + 聚类 + LLM 总结/待办")
    p.add_argument("--since", default="7d", help="抓取起始时间，示例: 7d / 48h / 2025-01-01")
    p.add_argument("--mailbox", default=None, help="IMAP 邮箱文件夹，默认使用配置 MAILBOX")
    p.add_argument("--output", default=None, help="输出 Markdown 文件路径，不填仅打印控制台")
    p.add_argument("--instruction", default="根据邮件生成我的待办并按优先级排序", help="给 LLM 的自然语言指令")
    p.add_argument("--rescan", action="store_true", help="忽略增量状态，重新抓取满足条件的邮件")
    p.add_argument("--include-seen", action="store_true", help="包含已读邮件")
    p.add_argument("--mode", choices=list(_MODE_INSTRUCTIONS.keys()), default=None,
                   help="预设模式：morning-briefing / follow-up-check / weekly-report")
    p.add_argument("--days", type=int, default=3,
                   help="--mode follow-up-check 时使用：超过 N 天未回复视为待跟进（默认 3）")
    p.add_argument("--session", default="default", help="对话 session ID，同 ID 下保留多轮上下文")
    p.add_argument("--json", action="store_true", dest="output_json",
                   help="Agent 模式下输出结构化 JSON 而非渲染 Markdown")
    p.add_argument("--chat", action="store_true",
                   help="交互式多轮对话模式：程序不退出，持续等待输入，邮件只拉取一次")

    args = p.parse_args(argv)

    config = load_config()
    mailbox = args.mailbox or config.mailbox

    # --mode 展开为 instruction
    instruction = args.instruction
    if args.mode:
        instruction = _MODE_INSTRUCTIONS[args.mode]
        if args.mode == "follow-up-check":
            instruction = instruction.replace("所有", f"超过 {args.days} 天")

    state = StateStore(config.state_path)
    fetcher = IMAPFetcher(config, state)

    if args.chat:
        return _run_chat_mode(fetcher, config, instruction, args)
    return _run_agent_mode(fetcher, config, instruction, args)


def _run_chat_mode(fetcher, config, first_instruction: str, args) -> int:
    """交互式多轮对话模式。

    - 第一轮：拉取邮件、聚类、执行指令
    - 后续轮：LangGraph 从 checkpoint 恢复 email_items/threads，fetch_node 自动跳过重拉
    - 输入 exit / quit / q 退出
    """
    from .agent.graph import build_graph
    from .email_store import load_store

    store = load_store(config)
    graph = build_graph(fetcher, config, store)

    session_id = args.session
    graph_config = {"configurable": {"thread_id": session_id}}

    print("=" * 60)
    print(f"[Chat] 交互式对话模式启动（session={session_id}）")
    print("[Chat] 邮件将在第一次指令时拉取，后续对话复用同一批邮件")
    print("[Chat] 输入 exit / quit / q 退出")
    print("=" * 60)

    # 第一轮使用完整初始状态（触发 fetch_node 拉邮件）
    initial_state = {
        "instruction": first_instruction,
        "rewritten_instruction": "",
        "sub_queries": [],
        "retrieved_context": "",
        "email_items": [],
        "threads": [],
        "output": None,
        "actions_taken": [],
        "messages": [],
        "critique": "",
        "reflection_count": 0,
        "enable_reflection": False,
    }

    instruction = first_instruction
    is_first_turn = True

    while True:
        if not is_first_turn:
            # 后续轮：只更新必要字段，不包含 messages（让 checkpoint 自然恢复历史）
            # email_items / threads 由 checkpoint 自动恢复，fetch_node 看到非空会跳过拉取
            input_state = {
                "instruction": instruction,
                "rewritten_instruction": "",
                "sub_queries": [],
                "retrieved_context": "",
                "output": None,
                "actions_taken": [],
                "critique": "",
                "reflection_count": 0,
            }
        else:
            input_state = initial_state

        print(f"\n[Agent] 处理中：{instruction}\n")

        for _ in graph.stream(input_state, config=graph_config, stream_mode="values"):
            pass

        final_state = graph.get_state(graph_config)
        output = final_state.values.get("output")

        if output is None:
            print("[Agent] 未生成输出\n")
        elif args.output_json:
            print(output.model_dump_json(indent=2))
        else:
            body = _agent_output_to_markdown(output)
            doc = render_document("邮件助理报告", body)
            print(doc)

        is_first_turn = False

        # 等待下一条用户输入
        print("-" * 60)
        try:
            raw = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Chat] 已退出")
            break

        if raw.lower() in ("exit", "quit", "q", ""):
            print("[Chat] 已退出")
            break

        instruction = raw

    return 0


def _run_agent_mode(fetcher, config, instruction: str, args) -> int:
    """通过 LangGraph Agent 图处理邮件。"""
    from .agent.graph import build_graph
    from .email_store import load_store

    store = load_store(config)
    graph = build_graph(fetcher, config, store)

    session_id = args.session
    graph_config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "instruction": instruction,
        "rewritten_instruction": "",
        "sub_queries": [],
        "retrieved_context": "",
        "email_items": [],
        "threads": [],
        "output": None,
        "actions_taken": [],
        "messages": [],
        "critique": "",
        "reflection_count": 0,
        "enable_reflection": False,
    }

    print(f"[Agent] 模式启动，session={session_id}，指令：{instruction}\n")

    # 流式执行直到完成
    for _ in graph.stream(initial_state, config=graph_config, stream_mode="values"):
        pass

    # 获取最终状态
    final_state = graph.get_state(graph_config)
    output = final_state.values.get("output")

    if output is None:
        print("[Agent] 未生成输出")
        return 1

    if args.output_json:
        result_json = output.model_dump_json(indent=2)
        print(result_json)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result_json)
    else:
        body = _agent_output_to_markdown(output)
        doc = render_document("邮件助理报告", body)
        print(doc)
        if args.output:
            write_markdown(doc, args.output)

    return 0


def _agent_output_to_markdown(output) -> str:
    """将 AgentOutput 转换为 Markdown 字符串用于渲染。"""
    lines = []

    if output.summary:
        lines += ["## 总结", output.summary, ""]

    if output.vip_alerts:
        lines += ["## ⭐ VIP 邮件"]
        for alert in output.vip_alerts:
            lines.append(f"- {alert}")
        lines.append("")

    if output.todos:
        lines += ["## 待办事项"]
        for todo in output.todos:
            deadline = f"（截止 {todo.deadline}）" if todo.deadline else ""
            action = f" → {todo.suggested_action}" if todo.suggested_action else ""
            lines.append(f"- **[{todo.priority}]** {todo.title}{deadline}{action}")
        lines.append("")

    if output.meetings:
        lines += ["## 会议日程"]
        for m in output.meetings:
            time_str = f" | 时间：{m.time}" if m.time else ""
            loc_str = f" | 地点：{m.location}" if m.location else ""
            lines.append(f"- **{m.subject}**{time_str}{loc_str}")
        lines.append("")

    if output.follow_ups:
        lines += ["## 跟进事项"]
        for fu in output.follow_ups:
            direction = "我方发起" if fu.direction == "sent" else "对方承诺"
            due = f"（{fu.due_date}）" if fu.due_date else ""
            lines.append(f"- [{direction}] {fu.description}{due}")
        lines.append("")

    if output.draft_replies:
        lines += ["## 草拟回复（待确认）"]
        for dr in output.draft_replies:
            lines.append(f"### 回复 UID={dr.to_uid}（语气：{dr.tone}）")
            lines.append(f"```\n{dr.draft}\n```")
        lines.append("")

    if output.unresolved:
        lines += ["## 待确认事项"]
        for u in output.unresolved:
            lines.append(f"- {u}")
        lines.append("")

    return "\n".join(lines)



if __name__ == "__main__":
    sys.exit(main())


