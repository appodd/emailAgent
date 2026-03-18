"""LangGraph Tool 定义 — Phase 4

工具分三组：
  · 读取类（无副作用）
  · 生成类（输出草稿，不直接发送）
  · 写操作（有副作用，需人工确认）
"""
from __future__ import annotations

from langchain_core.tools import tool


# ── 读取类（无副作用）────────────────────────────────────────────────────────

@tool
def fetch_email_detail(uid: int) -> str:
    """获取指定 UID 邮件的完整正文"""
    # 实现在 nodes.py 中通过闭包注入实际的 IMAPFetcher；
    # 此处作为工具签名定义供 LLM 感知。
    raise NotImplementedError("需通过 build_tools(fetcher, store, config) 注入实现")


@tool
def search_emails_by_topic(query: str) -> str:
    """在历史 + 本次邮件中语义搜索相关内容，返回摘要列表"""
    raise NotImplementedError("需通过 build_tools(fetcher, store, config) 注入实现")


@tool
def check_pending_replies(days: int = 3) -> str:
    """检查我方发出但超过 days 天未收到回复的邮件，返回待跟进列表"""
    raise NotImplementedError("需通过 build_tools(fetcher, store, config) 注入实现")


@tool
def extract_meeting_info(uid: int) -> str:
    """从指定邮件中提取会议时间、地点、参与者等结构化信息"""
    raise NotImplementedError("需通过 build_tools(fetcher, store, config) 注入实现")


# ── 生成类（输出草稿，不直接执行）────────────────────────────────────────────

@tool
def draft_reply(uid: int, instruction: str) -> str:
    """根据邮件内容和 instruction 指令起草回复正文，不自动发送，需人工确认"""
    raise NotImplementedError("需通过 build_tools(fetcher, store, config) 注入实现")


@tool
def draft_new_email(to: str, subject: str, instruction: str) -> str:
    """起草一封全新邮件（非回复）。to: 收件人，instruction: 写作要求，需人工确认后发送"""
    raise NotImplementedError("需通过 build_tools(fetcher, store, config) 注入实现")


@tool
def adjust_tone(draft: str, tone: str) -> str:
    """调整邮件草稿的语气。tone 可取 'formal'（正式）/ 'casual'（轻松）/ 'concise'（简洁）"""
    raise NotImplementedError("需通过 build_tools(fetcher, store, config) 注入实现")


@tool
def generate_report(period: str) -> str:
    """基于 RAG 历史邮件向量库生成汇总报告。period: 'daily' / 'weekly' / 'monthly'"""
    raise NotImplementedError("需通过 build_tools(fetcher, store, config) 注入实现")


# ── 写操作（有副作用，均需人工确认）─────────────────────────────────────────

@tool
def send_email(uid: int, to: str, subject: str, body: str) -> str:
    """发送邮件。uid=-1 时为新邮件，否则为回复指定 uid 的邮件。需人工确认后执行。"""
    raise NotImplementedError("需通过 build_tools(fetcher, store, config) 注入实现")


@tool
def flag_email_as_important(uid: int) -> str:
    """将指定邮件标记为重要/星标"""
    raise NotImplementedError("需通过 build_tools(fetcher, store, config) 注入实现")


@tool
def archive_email(uid: int, folder: str = "Archive") -> str:
    """将指定邮件移动到归档文件夹"""
    raise NotImplementedError("需通过 build_tools(fetcher, store, config) 注入实现")


@tool
def create_todo(title: str, priority: str, deadline: str = "") -> str:
    """创建一条待办事项到本地待办列表。priority: P0/P1/P2"""
    raise NotImplementedError("需通过 build_tools(fetcher, store, config) 注入实现")


def build_tools(fetcher, store, config):
    """
    返回注入了实际实现的 LangChain tool 列表。
    fetcher: IMAPFetcher 实例
    store:   FAISS 向量库（可为 None）
    config:  Config 实例
    """
    from datetime import datetime, timezone, timedelta
    from langchain_core.tools import tool as _tool
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=config.deepseek_model,
        base_url=config.deepseek_base_url,
        api_key=config.deepseek_api_key,
        temperature=0.3,
    )

    @_tool
    def fetch_email_detail(uid: int) -> str:
        """获取指定 UID 邮件的完整正文"""
        # 直接从缓存的 email_items 里找
        return f"[fetch_email_detail uid={uid}] 请在 plan_node 的提示词中提供邮件正文"

    @_tool
    def search_emails_by_topic(query: str) -> str:
        """在历史 + 本次邮件中语义搜索相关内容，返回摘要列表"""
        if store is None:
            return "向量库尚未建立，请先运行一次以建立历史索引"
        docs = store.similarity_search(query, k=5)
        results = []
        for i, d in enumerate(docs, 1):
            meta = d.metadata
            results.append(
                f"{i}. [{meta.get('date', '?')}] From: {meta.get('from', '?')} "
                f"| Subject: {meta.get('subject', '?')}\n   {d.page_content[:200]}"
            )
        return "\n\n".join(results) if results else "未找到相关历史邮件"

    @_tool
    def check_pending_replies(days: int = 3) -> str:
        """检查我方发出但超过 days 天未收到回复的邮件，返回待跟进列表"""
        if store is None:
            return "向量库尚未建立，无法检查历史发件"
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        docs = store.similarity_search("reply follow up pending", k=50)
        pending = []
        for d in docs:
            meta = d.metadata
            from_addr = meta.get("from", "")
            # 简单启发：发件地址包含自身用户名视为"我方发送"
            date_str = meta.get("date", "")
            try:
                doc_date = datetime.fromisoformat(date_str)
                if doc_date < cutoff:
                    pending.append(
                        f"- [{date_str}] To: {meta.get('subject', '?')} (from {from_addr})"
                    )
            except Exception:
                continue
        return "\n".join(pending[:10]) if pending else f"近 {days} 天内无明显待跟进邮件"

    @_tool
    def extract_meeting_info(uid: int) -> str:
        """从指定邮件中提取会议时间、地点、参与者等结构化信息"""
        prompt = (
            f"请从以下邮件(UID={uid})中提取会议信息（时间、地点、参与者、议程），"
            "以 JSON 格式返回。若无则返回空 JSON {}。"
        )
        resp = llm.invoke(prompt)
        return resp.content

    @_tool
    def draft_reply(uid: int, instruction: str) -> str:
        """根据邮件内容和 instruction 指令起草回复正文，不自动发送，需人工确认"""
        prompt = (
            f"请根据以下指令为 UID={uid} 的邮件起草回复草稿：\n{instruction}\n\n"
            "仅输出回复正文，不要加客套话开头。"
        )
        resp = llm.invoke(prompt)
        return resp.content

    @_tool
    def draft_new_email(to: str, subject: str, instruction: str) -> str:
        """起草一封全新邮件（非回复）。to: 收件人，instruction: 写作要求，需人工确认后发送"""
        prompt = (
            f"请起草一封邮件：\n收件人: {to}\n主题: {subject}\n写作要求: {instruction}\n\n"
            "仅输出邮件正文。"
        )
        resp = llm.invoke(prompt)
        return resp.content

    @_tool
    def adjust_tone(draft: str, tone: str) -> str:
        """调整邮件草稿的语气。tone 可取 'formal'（正式）/ 'casual'（轻松）/ 'concise'（简洁）"""
        tone_map = {"formal": "正式商务", "casual": "轻松友好", "concise": "简洁精炼"}
        tone_cn = tone_map.get(tone, tone)
        prompt = f"请将以下邮件草稿调整为{tone_cn}语气，保持原意：\n\n{draft}"
        resp = llm.invoke(prompt)
        return resp.content

    @_tool
    def generate_report(period: str) -> str:
        """基于 RAG 历史邮件向量库生成汇总报告。period: 'daily' / 'weekly' / 'monthly'"""
        period_map = {"daily": "今日", "weekly": "本周", "monthly": "本月"}
        period_cn = period_map.get(period, period)
        if store is None:
            return "向量库尚未建立，无法生成报告"
        docs = store.similarity_search("summary report", k=20)
        context = "\n\n".join(
            f"[{d.metadata.get('date', '?')}] {d.page_content[:300]}" for d in docs[:10]
        )
        prompt = (
            f"请基于以下历史邮件内容，生成一份{period_cn}邮件汇总报告，"
            "包括主要事项、跟进项、决策点：\n\n" + context
        )
        resp = llm.invoke(prompt)
        return resp.content

    @_tool
    def send_email(uid: int, to: str, subject: str, body: str) -> str:
        """发送邮件。uid=-1 时为新邮件，否则为回复指定 uid 的邮件。需人工确认后执行。"""
        # 实际 SMTP 发送逻辑留给后续实现
        action = "回复" if uid >= 0 else "新邮件"
        return f"[待确认] 将{action}发送至 {to}，主题：{subject}。正文已准备好，请确认后执行。"

    @_tool
    def flag_email_as_important(uid: int) -> str:
        """将指定邮件标记为重要/星标"""
        return f"[待执行] UID={uid} 的邮件已标记为重要（实际 IMAP 标记需后续实现）"

    @_tool
    def archive_email(uid: int, folder: str = "Archive") -> str:
        """将指定邮件移动到归档文件夹"""
        return f"[待执行] UID={uid} 的邮件将移至 {folder}（实际 IMAP 移动需后续实现）"

    @_tool
    def create_todo(title: str, priority: str, deadline: str = "") -> str:
        """创建一条待办事项到本地待办列表。priority: P0/P1/P2"""
        import json, os
        todo_path = "todos.json"
        todos = []
        if os.path.exists(todo_path):
            with open(todo_path, "r", encoding="utf-8") as f:
                todos = json.load(f)
        entry = {"title": title, "priority": priority, "deadline": deadline, "done": False}
        todos.append(entry)
        with open(todo_path, "w", encoding="utf-8") as f:
            json.dump(todos, f, ensure_ascii=False, indent=2)
        return f"已创建待办：[{priority}] {title}" + (f"（截止：{deadline}）" if deadline else "")

    return [
        fetch_email_detail,
        search_emails_by_topic,
        check_pending_replies,
        extract_meeting_info,
        draft_reply,
        draft_new_email,
        adjust_tone,
        generate_report,
        send_email,
        flag_email_as_important,
        archive_email,
        create_todo,
    ]
