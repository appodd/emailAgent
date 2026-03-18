"""LangGraph 图节点 — Phase 4 + Phase 5 + Phase 6"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..models import EmailItem
from .state import AgentState
from .structured import AgentOutput

if TYPE_CHECKING:
    from ..config import Config
    from ..imap_client import IMAPFetcher


# ──────────────────────────────────────────────────────────────────────────────
# 工厂函数：注入依赖后返回节点函数
# ──────────────────────────────────────────────────────────────────────────────

def make_nodes(fetcher: "IMAPFetcher", config: "Config", tools: list):
    """
    返回 (fetch_node, cluster_node, plan_node, act_node, summarize_node, router)
    通过闭包将 fetcher/config/tools 注入，避免全局状态。
    """
    llm = ChatOpenAI(
        model=config.deepseek_model,
        base_url=config.deepseek_base_url,
        api_key=config.deepseek_api_key,
        temperature=0.2,
    )
    llm_with_tools = llm.bind_tools(tools)
    structured_llm = llm.with_structured_output(AgentOutput, method="function_calling")

    # ── 节点 1：fetch_node ────────────────────────────────────────────────────

    def fetch_node(state: AgentState) -> dict:
        """从 IMAP 抓取邮件，若 state 已有 email_items 则跳过抓取。
        Phase 6: 同时进行 RAG 历史上下文注入。
        """
        if state.get("email_items"):
            return {}  # 已有数据，跳过

        from datetime import timezone, timedelta
        from ..state_store import StateStore
        from ..email_store import build_or_update_store, load_store, build_hybrid_retriever, expand_with_parents

        since = datetime.now(timezone.utc) - timedelta(days=7)
        state_store = StateStore(config.state_path)
        new_items: List[EmailItem] = fetcher.fetch_unread(since, ignore_state=False)

        # Phase 6: RAG 历史邮件上下文注入
        store = load_store(config)
        if store and new_items:
            try:
                all_docs = list(store.docstore._dict.values())
                retriever = build_hybrid_retriever(all_docs, store)
                msg_id_index = {
                    d.metadata["message_id"]: d
                    for d in all_docs
                    if d.metadata.get("message_id")
                }
                for item in new_items:
                    ref_ids = (item.references or "").split()
                    parent_docs = [msg_id_index[rid] for rid in ref_ids if rid in msg_id_index]
                    if parent_docs:
                        parent_docs.sort(key=lambda d: d.metadata.get("date", ""))
                        item.historical_context = "\n---\n".join(d.page_content for d in parent_docs)
                    else:
                        query = f"{item.from_addr} {item.subject}"
                        retrieved = retriever.invoke(query)
                        expanded = expand_with_parents(retrieved, msg_id_index, max_depth=3)
                        item.historical_context = "\n---\n".join(d.page_content for d in expanded)
            except Exception as e:
                # RAG 失败不阻断主流程
                print(f"[fetch_node] RAG 上下文注入失败: {e}")

        # 写入向量库（供后续检索）
        if new_items:
            try:
                build_or_update_store(new_items, config)
            except Exception as e:
                print(f"[fetch_node] 向量库更新失败: {e}")

        return {"email_items": new_items}

    # ── 节点 2：cluster_node ──────────────────────────────────────────────────

    def cluster_node(state: AgentState) -> dict:
        """Embedding 聚类（Phase 3）或 TF-IDF 兜底"""
        items = state.get("email_items", [])
        if not items:
            return {"threads": []}

        from ..clusterer import cluster_emails
        result = cluster_emails(items, config)
        return {"threads": result.threads}

    # ── 节点 2.5：retrieve_node ───────────────────────────────────────────────

    def retrieve_node(state: AgentState) -> dict:
        """基于 rewrite_node 产出的 sub_queries 做多路并行 RAG 检索。

        · 若 sub_queries 为空或向量库不存在，则跳过，retrieved_context 保持空字符串
        · 多路检索结果按 uid 去重合并，限制总长度避免 prompt 过长
        """
        from ..email_store import load_store, build_hybrid_retriever, expand_with_parents

        sub_queries = state.get("sub_queries", [])
        if not sub_queries:
            return {"retrieved_context": ""}

        store = load_store(config)
        if store is None:
            return {"retrieved_context": ""}

        try:
            all_docs = list(store.docstore._dict.values())
            retriever = build_hybrid_retriever(all_docs, store, config)
            msg_id_index = {
                d.metadata["message_id"]: d
                for d in all_docs
                if d.metadata.get("message_id")
            }

            # 多路检索并合并去重
            docs = retriever.invoke_multi(sub_queries)
            # 父链展开，还原完整对话链
            docs = expand_with_parents(docs, msg_id_index, max_depth=2)

            # 格式化为文本，每条限 300 字符，最多 5 条
            snippets = []
            for d in docs[:5]:
                meta = d.metadata
                snippets.append(
                    f"[{meta.get('date', '?')}] From: {meta.get('from', '?')} "
                    f"| Subject: {meta.get('subject', '?')}\n{d.page_content[:300]}"
                )
            retrieved_context = "\n\n---\n\n".join(snippets)
            print(f"[RetrieveNode] sub_queries={sub_queries}, 召回 {len(docs)} 条")
        except Exception as e:
            print(f"[RetrieveNode] 检索失败: {e}")
            retrieved_context = ""

        return {"retrieved_context": retrieved_context}

    # ── 节点 3：plan_node ─────────────────────────────────────────────────────

    def plan_node(state: AgentState) -> dict:
        """LLM 分析线程，规划下一步动作（调用工具 or 直接总结）"""
        threads = state.get("threads", [])
        # 优先使用 rewrite_node 改写后的指令，首轮退回原始指令
        instruction = state.get("rewritten_instruction") or state.get("instruction", "请处理这些邮件")
        critique = state.get("critique", "")
        retrieved_context = state.get("retrieved_context", "")

        # 构造线程摘要文本
        thread_blocks = []
        for t in threads[:20]:  # 最多 20 个线程送入 prompt
            items_text = []
            for it in t.items[-4:]:
                ctx = f"\n  [历史上下文]:\n  {it.historical_context[:400]}" if it.historical_context else ""
                items_text.append(
                    f"  - [{it.date.isoformat() if isinstance(it.date, datetime) else it.date}] "
                    f"From: {it.from_addr}\n  {(it.text or '')[:600]}{ctx}"
                )
            thread_blocks.append(f"【{t.subject_fingerprint}】\n" + "\n".join(items_text))

        threads_text = "\n\n".join(thread_blocks) or "（无邮件线程）"

        sys_content = (
            "你是一名全功能邮件助理，能够整理待办、提取会议、起草回复、追踪跟进、生成报告。"
            "根据用户指令和邮件内容，决定是否需要调用工具获取更多信息，"
            "或直接基于现有信息生成结构化输出。\n"
            "重要规则：\n"
            "1. 如果用户追问某封邮件的具体细节（如截止时间、具体要求、联系方式等），"
            "必须先调用 fetch_email_detail 工具获取完整邮件正文，再作答，不要凭摘要猜测。\n"
            "2. 如果用户问题涉及'什么时候''几号''deadline'等时间类追问，"
            "必须调用 fetch_email_detail 或 search_emails_by_topic 确认原文后再回答。\n"
            "3. 多轮对话中，你可以参考之前的对话记录来理解用户的上下文。"
        )

        critique_block = f"\n\n【上轮反思意见，请在本轮修正】：{critique}" if critique else ""
        retrieved_block = f"\n\n【RAG 检索到的相关历史邮件】：\n{retrieved_context}" if retrieved_context else ""

        user_content = (
            f"用户指令：{instruction}{critique_block}{retrieved_block}\n\n"
            f"邮件线程（共 {len(threads)} 个）：\n\n{threads_text}"
        )

        messages = state.get("messages", [])
        if not messages:
            messages = [
                SystemMessage(content=sys_content),
                HumanMessage(content=user_content),
            ]
        else:
            messages = list(messages) + [HumanMessage(content=user_content)]

        response = llm_with_tools.invoke(messages)
        return {
            "messages": messages + [response],
            "critique": "",  # 清空上轮 critique
        }

    # ── 节点 4：act_node ──────────────────────────────────────────────────────

    def act_node(state: AgentState) -> dict:
        """执行工具调用（ReAct 循环）"""
        from langchain_core.messages import ToolMessage
        messages = list(state.get("messages", []))
        last_msg = messages[-1]
        tool_map = {t.name: t for t in tools}

        tool_messages = []
        actions_taken = list(state.get("actions_taken", []))

        for tc in last_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"[错误] {e}"
            else:
                result = f"[未知工具] {tool_name}"

            actions_taken.append(f"{tool_name}({tool_args}) → {str(result)[:100]}")
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

        return {
            "messages": messages + tool_messages,
            "actions_taken": actions_taken,
        }

    # ── 节点 5：summarize_node ────────────────────────────────────────────────

    def summarize_node(state: AgentState) -> dict:
        """生成最终结构化输出（AgentOutput）"""
        threads = state.get("threads", [])
        instruction = state.get("instruction", "")
        actions_taken = state.get("actions_taken", [])

        thread_blocks = []
        for t in threads[:20]:
            items_text = []
            for it in t.items[-4:]:
                items_text.append(
                    f"  - [{it.date.isoformat() if isinstance(it.date, datetime) else it.date}] "
                    f"From: {it.from_addr} | {(it.text or '')[:300]}"
                )
            thread_blocks.append(f"【{t.subject_fingerprint}】\n" + "\n".join(items_text))

        threads_text = "\n\n".join(thread_blocks)
        actions_text = "\n".join(actions_taken) if actions_taken else "无"

        sys_content = (
            "你是一名全功能邮件助理，输出严格符合 AgentOutput 结构的 JSON。"
            "todos: 待办事项列表；meetings: 会议信息；follow_ups: 跟进事项；"
            "draft_replies: 草拟回复；vip_alerts: VIP 邮件摘要；summary: 整体摘要；unresolved: 待确认项。"
        )
        user_content = (
            f"用户指令：{instruction}\n\n"
            f"已执行动作：\n{actions_text}\n\n"
            f"邮件线程：\n{threads_text}"
        )

        try:
            output: AgentOutput = structured_llm.invoke([
                SystemMessage(content=sys_content),
                HumanMessage(content=user_content),
            ])
        except Exception as e:
            # 降级：返回最小输出
            output = AgentOutput(
                todos=[],
                summary=f"结构化输出失败：{e}。已执行：{actions_text}",
            )

        return {"output": output}

    # ── 节点 6：reflect_node（F1，可选）──────────────────────────────────────

    def reflect_node(state: AgentState) -> dict:
        """LLM 自评估当前输出质量，返回 critique 或空字符串"""
        output = state.get("output")
        if output is None:
            return {"critique": "", "reflection_count": state.get("reflection_count", 0) + 1}

        critique_prompt = (
            f"你刚刚生成了以下邮件处理结果：\n{output.model_dump_json(indent=2)}\n\n"
            "请检查：\n"
            "1. 是否有紧急邮件未被纳入 P0 待办？\n"
            "2. 草拟回复的语气是否符合上下文？\n"
            "3. 是否遗漏了需要跟进的承诺事项？\n\n"
            "如果发现问题，请简述 critique；如果质量满足，直接回复「OK」。"
        )
        resp = llm.invoke(critique_prompt)
        critique = resp.content.strip()
        if critique.upper() == "OK":
            critique = ""
        return {
            "critique": critique,
            "reflection_count": state.get("reflection_count", 0) + 1,
        }

    # ── 节点 Q：rewrite_node ──────────────────────────────────────────────────

    # 触发 LLM 改写的指代词列表
    _PRONOUN_TRIGGERS = {"他", "她", "它", "这封", "那个", "那封", "上一封", "刚才", "之前", "那次", "这个"}
    # 触发子查询分解的关键词
    _MULTI_INTENT_TRIGGERS = {"比较", " vs ", " VS ", "和", "与", "区别", "对比"}

    def _needs_rewrite(instruction: str, messages: list) -> bool:
        """意图路由：判断是否需要 LLM 改写。满足任一条件则需要改写。"""
        # 有对话历史
        if messages:
            return True
        # 包含指代词
        if any(t in instruction for t in _PRONOUN_TRIGGERS):
            return True
        # 包含多实体比较关键词
        if any(t in instruction for t in _MULTI_INTENT_TRIGGERS):
            return True
        # 过短且无实体（可能省略了主语）
        if len(instruction) < 15:
            return True
        return False

    def rewrite_node(state: AgentState) -> dict:
        """基于对话历史对用户指令做 Query Rewrite（P0 增强版）。

        · 意图路由：简单明确的首轮指令直接透传，不消耗 LLM token
        · 上下文补全：多轮历史脱水，补全代词/省略，形成自包含查询
        · 子查询分解：多实体/比较类查询拆解为子查询列表，供并行 RAG 检索
        """
        from .structured import RewriteOutput

        instruction = state.get("instruction", "")
        messages = state.get("messages", [])

        # ── 意图路由：不需要改写则直接透传 ──────────────────────────────────
        if not _needs_rewrite(instruction, messages):
            print(f"[QueryRewrite] 意图路由：简单明确，透传 → {instruction}")
            return {
                "rewritten_instruction": instruction,
                "sub_queries": [instruction],
            }

        # ── 构造历史摘要（最近 6 条消息）────────────────────────────────────
        history_lines = []
        for msg in messages[-6:]:
            role = type(msg).__name__.replace("Message", "")
            content = getattr(msg, "content", "") or ""
            if content:
                history_lines.append(f"[{role}]: {str(content)[:200]}")
        history_text = "\n".join(history_lines) if history_lines else "（无历史）"

        rewrite_sys = (
            "你是一名查询改写助理，专门处理邮件助手场景下的用户指令。\n"
            "任务：根据对话历史，将用户最新指令改写为结构化输出。\n\n"
            "规则：\n"
            "1. rewritten_instruction：补全代词（他/她/这封/那个→具体实体），补全省略的时间/对象，"
            "保留原始意图，不添加不存在的假设。若原始指令已足够清晰则原样返回。\n"
            "2. sub_queries：将检索意图拆解为独立子查询列表。\n"
            "   - 单一意图（如'总结张三的最新邮件'）→ 返回 ['张三 最新邮件']\n"
            "   - 多实体/比较意图（如'比较A项目和B项目进展'）→ 返回 ['A项目进展', 'B项目进展']\n"
            "   - 子查询应该是简洁的名词短语，不要带'帮我'/'请'等命令词\n"
            "3. needs_rewrite：若对原始指令做了实质修改则为 true，否则为 false。\n"
        )
        rewrite_human = (
            f"对话历史：\n{history_text}\n\n"
            f"用户最新指令：{instruction}\n\n"
            "请按照格式输出改写结果："
        )

        rewrite_llm = llm.with_structured_output(RewriteOutput, method="function_calling")

        try:
            result: RewriteOutput = rewrite_llm.invoke([
                SystemMessage(content=rewrite_sys),
                HumanMessage(content=rewrite_human),
            ])
            rewritten = result.rewritten_instruction.strip() or instruction
            sub_queries = [q.strip() for q in result.sub_queries if q.strip()]
            if not sub_queries:
                sub_queries = [rewritten]
        except Exception as e:
            print(f"[QueryRewrite] 改写失败，使用原始指令: {e}")
            rewritten = instruction
            sub_queries = [instruction]

        print(f"[QueryRewrite] 原始:     {instruction}")
        print(f"[QueryRewrite] 改写:     {rewritten}")
        print(f"[QueryRewrite] 子查询:   {sub_queries}")
        return {
            "rewritten_instruction": rewritten,
            "sub_queries": sub_queries,
        }


    # ── 路由函数 ──────────────────────────────────────────────────────────────

    def router(state: AgentState) -> str:
        """条件路由：判断是否继续 act 还是进入 summarize"""
        messages = state.get("messages", [])
        if not messages:
            return "summarize"
        last_msg = messages[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "act"
        return "summarize"

    def reflection_router(state: AgentState) -> str:
        """根据 reflect_node 的 critique 决定是否重新推理"""
        critique = state.get("critique", "")
        reflection_count = state.get("reflection_count", 0)
        enable = state.get("enable_reflection", False)
        if enable and critique and reflection_count < 2:
            return "plan"
        return "__end__"

    return fetch_node, cluster_node, retrieve_node, rewrite_node, plan_node, act_node, summarize_node, reflect_node, router, reflection_router
