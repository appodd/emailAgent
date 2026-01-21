from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Config
from .models import EmailItem, EmailThread, ClusterResult
from .text_utils import normalize_subject, email_address_domain


def _time_bucket(dt: datetime, window_hours: int) -> int:
    epoch = int(dt.replace(tzinfo=dt.tzinfo or timezone.utc).timestamp())
    size = window_hours * 3600
    return epoch // size


def pre_bucket(items: List[EmailItem], config: Config) -> Dict[str, List[EmailItem]]:
    buckets: Dict[str, List[EmailItem]] = defaultdict(list)
    for it in items:
        subj = normalize_subject(it.subject)
        domain = email_address_domain(it.from_addr)
        tbin = _time_bucket(it.date, config.time_window_hours)
        key = f"{subj}|{domain}|{tbin}"
        buckets[key].append(it)
    return buckets


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _vectorize_texts(texts: List[str]) -> np.ndarray:
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    return vec.fit_transform(texts)


def refine_with_similarity(bucket_key: str, items: List[EmailItem], sim_threshold: float) -> List[EmailThread]:
    if len(items) == 1:
        it = items[0]
        return [EmailThread(
            thread_id=f"{bucket_key}#0",
            subject_fingerprint=normalize_subject(it.subject),
            participants=[it.from_addr] + it.to_addrs + it.cc_addrs,
            items=[it],
        )]

    corpus = [f"{normalize_subject(it.subject)}\n\n{(it.text or '')[:500]}" for it in items]
    X = _vectorize_texts(corpus)
    sim = cosine_similarity(X)
    n = len(items)
    uf = _UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= sim_threshold:
                uf.union(i, j)

    comp_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        comp_to_indices[uf.find(i)].append(i)

    threads: List[EmailThread] = []
    for k, idxs in comp_to_indices.items():
        grouped = [items[i] for i in sorted(idxs, key=lambda x: items[x].date)]
        participants = []
        seen = set()
        for it in grouped:
            for p in [it.from_addr] + it.to_addrs + it.cc_addrs:
                if p and p not in seen:
                    participants.append(p)
                    seen.add(p)
        subj_fp = normalize_subject(grouped[-1].subject)
        threads.append(
            EmailThread(
                thread_id=f"{bucket_key}#{k}",
                subject_fingerprint=subj_fp,
                participants=participants,
                items=grouped,
            )
        )
    return threads


def cluster_emails(items: List[EmailItem], config: Config) -> ClusterResult:
    buckets = pre_bucket(items, config)
    all_threads: List[EmailThread] = []
    for bkey, bucket_items in buckets.items():
        threads = refine_with_similarity(bkey, bucket_items, config.sim_threshold)
        all_threads.extend(threads)
    return ClusterResult(threads=sorted(all_threads, key=lambda t: t.items[-1].date, reverse=True))


