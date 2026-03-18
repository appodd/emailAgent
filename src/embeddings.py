"""Embedding 语义聚类 — Phase 3

替换 TF-IDF，使用 OpenAI Embedding + 层次聚类。
通过 config.enable_agent_mode 开关；False 时 clusterer.py 仍走旧 TF-IDF 路径。
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, List

import numpy as np

from .models import EmailItem, EmailThread, ClusterResult
from .text_utils import normalize_subject

if TYPE_CHECKING:
    from .config import Config


def embed_and_cluster(items: List[EmailItem], config: "Config") -> ClusterResult:
    """
    用 OpenAI Embedding 向量化邮件，再做层次聚类（AgglomerativeClustering）。
    不需要预先指定 k，通过 distance_threshold 自动决定分组数。
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from sklearn.cluster import AgglomerativeClustering

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # 用主题 + 正文前 500 字符作为向量化文本
    texts = [f"{normalize_subject(it.subject)}\n{(it.text or '')[:500]}" for it in items]

    # 批量获取向量
    vectors = embeddings_model.embed_documents(texts)
    vectors = np.array(vectors)

    if len(vectors) == 1:
        # 单封邮件直接返回
        it = items[0]
        return ClusterResult(threads=[
            EmailThread(
                thread_id="0",
                subject_fingerprint=normalize_subject(it.subject),
                participants=[it.from_addr] + it.to_addrs + it.cc_addrs,
                items=[it],
            )
        ])

    # 层次聚类（cosine 距离，不需要预设 k）
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.4,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(vectors)

    # 按 label 分组
    comp_to_indices: dict[int, List[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        comp_to_indices[int(label)].append(i)

    threads: List[EmailThread] = []
    for label, idxs in comp_to_indices.items():
        grouped = [items[i] for i in sorted(idxs, key=lambda x: items[x].date)]
        participants: List[str] = []
        seen: set[str] = set()
        for it in grouped:
            for p in [it.from_addr] + it.to_addrs + it.cc_addrs:
                if p and p not in seen:
                    participants.append(p)
                    seen.add(p)
        subj_fp = normalize_subject(grouped[-1].subject)
        threads.append(
            EmailThread(
                thread_id=str(label),
                subject_fingerprint=subj_fp,
                participants=participants,
                items=grouped,
            )
        )

    # 按最新邮件时间降序
    threads.sort(key=lambda t: t.items[-1].date, reverse=True)
    return ClusterResult(threads=threads)
