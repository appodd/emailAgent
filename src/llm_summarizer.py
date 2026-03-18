from __future__ import annotations

from typing import List

from .config import Config
from .models import EmailThread


def summarize_threads_structured(threads: List[EmailThread], instruction: str, config: Config):
    """返回 AgentOutput Pydantic 对象（Phase 2 结构化输出）。"""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from .agent.structured import AgentOutput

    if not threads:
        return AgentOutput(todos=[], summary="_无可总结的邮件线程。_")

    llm = ChatOpenAI(
        model=config.deepseek_model,
        base_url=config.deepseek_base_url,
        api_key=config.deepseek_api_key,
        temperature=0.2,
    )
    structured_llm = llm.with_structured_output(AgentOutput, method="function_calling")

    threads_block = "\n\n".join(
        f"主题指纹: {t.subject_fingerprint}\n" +
        "\n".join(
            f"  - [{it.date.isoformat() if hasattr(it.date, 'isoformat') else it.date}] "
            f"From: {it.from_addr} | {(it.text or '').strip()[:300]}"
            for it in t.items[-6:]
        )
        for t in threads
    )
    messages = [
        SystemMessage(content=(
            "你是一名全功能邮件助理，能整理待办、提取会议、起草回复、追踪跟进、生成报告。"
            "请基于邮件线程按用户指令生成结构化输出，优先级：P0=紧急、P1=重要、P2=一般。"
        )),
        HumanMessage(content=f"用户指令：{instruction}\n\n邮件线程：\n\n{threads_block}"),
    ]
    try:
        return structured_llm.invoke(messages)
    except Exception as e:
        return AgentOutput(todos=[], summary=f"结构化输出失败: {e}")


