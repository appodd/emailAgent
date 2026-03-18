"""RAG 历史邮件向量库 — Phase 6

功能：
  · strip_quoted_content     剥离引用内容，只保留新鲜正文
  · build_email_document     构建 FAISS Document（page_content=干净正文，metadata=线程关系）
  · build_or_update_store    增量写入/初始化 FAISS 向量库
  · load_store               加载本地向量库
  · expand_with_parents      检索后父链展开，还原完整对话链
  · build_hybrid_retriever   BM25 + FAISS + Cross-encoder Rerank 三级检索器
"""
from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Dict, List, Optional

from langchain_core.documents import Document

if TYPE_CHECKING:
    from .config import Config
    from .models import EmailItem


# ──────────────────────────────────────────────────────────────────────────────
# 文本预处理
# ──────────────────────────────────────────────────────────────────────────────

def strip_quoted_content(text: str) -> str:
    """剥离邮件中的引用历史内容，只保留新鲜正文。

    处理两种常见引用形式：
      1. > 开头的引用行
      2. "On ... wrote:" 分隔符（之后全部为旧内容）
    """
    lines = text.splitlines()
    fresh_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">"):
            continue
        if re.match(r"^On .+wrote:\s*$", stripped):
            break
        fresh_lines.append(line)
    return "\n".join(fresh_lines).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Document 构建
# ──────────────────────────────────────────────────────────────────────────────

def build_email_document(item: "EmailItem") -> Document:
    """
    构建入库 Document：
      · page_content = 剥离引用后的干净正文（负责 embedding/BM25 质量）
      · metadata     = 完整元信息 + 线程父子关系（负责检索后上下文还原）
    """
    fresh_text = strip_quoted_content(item.text or "")
    body = fresh_text[:1000] if fresh_text else (item.text or "")[:1000]

    return Document(
        page_content=body,
        metadata={
            "uid": item.uid,
            "subject": item.subject,
            "from": item.from_addr,
            "message_id": item.message_id or "",
            "in_reply_to": item.in_reply_to or "",
            "references": item.references or "",
            "date": item.date.isoformat() if hasattr(item.date, "isoformat") else str(item.date),
            "mailbox": item.mailbox,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# 向量库读写
# ──────────────────────────────────────────────────────────────────────────────

def _make_embeddings(config: "Config"):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def build_or_update_store(items: List["EmailItem"], config: "Config"):
    """将处理过的邮件增量写入 FAISS 向量库，持久化到本地。

    · 若本地已有向量库 → 追加（update），不重建
    · 否则 → 全量初始化（build）
    """
    from langchain_community.vectorstores import FAISS

    embeddings = _make_embeddings(config)
    docs = [build_email_document(it) for it in items]
    if not docs:
        return None

    store_path = config.email_store_path
    if os.path.exists(store_path):
        store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        store.add_documents(docs)
    else:
        store = FAISS.from_documents(docs, embeddings)

    store.save_local(store_path)
    return store


def load_store(config: "Config") -> Optional[object]:
    """从本地加载 FAISS 向量库；若不存在（首次运行）返回 None。"""
    from langchain_community.vectorstores import FAISS

    store_path = config.email_store_path
    if not os.path.exists(store_path):
        return None
    embeddings = _make_embeddings(config)
    return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)


# ──────────────────────────────────────────────────────────────────────────────
# 父链展开
# ──────────────────────────────────────────────────────────────────────────────

def expand_with_parents(
    retrieved_docs: List[Document],
    msg_id_index: Dict[str, Document],
    max_depth: int = 3,
) -> List[Document]:
    """检索后展开父链：对每条检索结果递归追加父邮件，还原完整对话链。

    Parameters
    ----------
    retrieved_docs : 初始检索结果
    msg_id_index   : message_id → Document 的映射（用于精确父链查找）
    max_depth      : 最大递归深度（防止死循环），默认 3

    Returns
    -------
    按日期升序排列的完整对话链文档列表
    """
    expanded = list(retrieved_docs)
    seen = {d.metadata.get("message_id") for d in expanded}
    queue = list(retrieved_docs)
    depth = 0

    while queue and depth < max_depth:
        next_queue: List[Document] = []
        for doc in queue:
            parent_id = doc.metadata.get("in_reply_to", "")
            if parent_id and parent_id not in seen and parent_id in msg_id_index:
                parent_doc = msg_id_index[parent_id]
                expanded.append(parent_doc)
                seen.add(parent_id)
                next_queue.append(parent_doc)
        queue = next_queue
        depth += 1

    expanded.sort(key=lambda d: d.metadata.get("date", ""))
    return expanded


# ──────────────────────────────────────────────────────────────────────────────
# 三级混合检索器
# ──────────────────────────────────────────────────────────────────────────────

class _HybridRetriever:
    """基于 RRF（Reciprocal Rank Fusion）的 BM25 + FAISS 混合检索器。

    不依赖 langchain.retrievers.EnsembleRetriever（该模块在 langchain 0.3 中已移除），
    直接实现 RRF 融合，兼容所有 langchain 版本。
    """

    def __init__(self, bm25_retriever, faiss_retriever, bm25_weight: float = 0.4,
                 rerank_model: Optional[str] = None, rerank_top_n: int = 3):
        self.bm25 = bm25_retriever
        self.faiss = faiss_retriever
        self.bm25_weight = bm25_weight
        self._reranker = None
        self._rerank_top_n = rerank_top_n
        if rerank_model:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(rerank_model)
            except Exception:
                pass  # 模型未下载，降级为 RRF 结果

    def _rrf_merge(self, lists: List[List[Document]], k: int = 60) -> List[Document]:
        """Reciprocal Rank Fusion：将多个排名列表合并为单一得分列表。"""
        scores: Dict[str, float] = {}
        id_to_doc: Dict[str, Document] = {}
        for rank_list in lists:
            for rank, doc in enumerate(rank_list):
                doc_id = doc.metadata.get("uid", doc.page_content[:80])
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
                id_to_doc[doc_id] = doc
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [id_to_doc[i] for i in sorted_ids]

    def invoke(self, query: str) -> List[Document]:
        bm25_docs = self.bm25.invoke(query)
        faiss_docs = self.faiss.invoke(query)
        merged = self._rrf_merge([bm25_docs, faiss_docs])
        if self._reranker and merged:
            pairs = [(query, d.page_content) for d in merged]
            scores = self._reranker.predict(pairs)
            ranked = sorted(zip(scores, merged), key=lambda x: x[0], reverse=True)
            return [d for _, d in ranked[: self._rerank_top_n]]
        return merged

    def invoke_multi(self, queries: List[str]) -> List[Document]:
        """多路子查询并行检索，结果按 uid 去重后返回。

        每条 sub_query 独立经过 BM25+FAISS+RRF，最终合并去重，
        保留各路召回结果的并集，避免单一查询遗漏另一侧实体。
        """
        seen_uids: set = set()
        result: List[Document] = []
        for q in queries:
            docs = self.invoke(q)
            for doc in docs:
                uid = doc.metadata.get("uid")
                key = uid if uid is not None else doc.page_content[:80]
                if key not in seen_uids:
                    seen_uids.add(key)
                    result.append(doc)
        return result

    # 兼容 LangChain Runnable 接口
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)


def build_hybrid_retriever(docs: List[Document], vectorstore, config: "Config" = None):
    """构建 BM25 + FAISS 混合检索 + Cross-encoder Rerank 的三级检索器。

    Step 1：BM25（精确词匹配）+ FAISS 向量检索，RRF 融合 → Top-20 粗召回
    Step 2：Cross-encoder Rerank → Top-3 精召回（需要 sentence-transformers 模型）
    """
    from langchain_community.retrievers import BM25Retriever

    rerank_top_n = config.rag_rerank_top_n if config else 3
    rerank_model = config.rag_rerank_model if config else None  # 默认不启用，需显式配置
    bm25_weight = config.rag_bm25_weight if config else 0.4

    bm25 = BM25Retriever.from_documents(docs, k=20)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    return _HybridRetriever(
        bm25_retriever=bm25,
        faiss_retriever=faiss_retriever,
        bm25_weight=bm25_weight,
        rerank_model=rerank_model,
        rerank_top_n=rerank_top_n,
    )
