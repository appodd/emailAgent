# LangChain Agent 改造计划

> **项目**：emailAgent  
> **目标**：将现有线性流水线升级为基于 LangGraph 的 Agent 架构，打造一个**全功能邮件助理**——不仅能整理待办、提取会议，还能起草回复、追踪跟进、生成报告，以多轮对话方式响应任意邮件处理需求  
> **日期**：2026-03-02

---

## 一、现状分析

### 当前架构（固定线性流水线）

```
CLI 入口 (cli.py)
    │
    ├─► IMAPFetcher.fetch_unread()      # 抓取邮件
    │
    ├─► cluster_emails()                # TF-IDF + cosine 聚类
    │
    └─► summarize_threads()             # 单次 LLM 调用 → Markdown 输出
```

### 核心痛点

| 问题 | 具体表现 |
|---|---|
| LLM 是被动总结器 | 无法采取任何行动，只能输出文字 |
| 流程完全固定 | 无法根据邮件内容动态决策 |
| TF-IDF 语义弱 | 跨语言或语义相近但措辞不同的邮件无法正确聚类 |
| 输出非结构化 | 返回原始 Markdown 字符串，难以对接下游系统 |
| 无状态交互 | 每次运行互相独立，无法追问或多轮对话 |

---

## 二、目标架构

```
用户指令
    │
    ▼
┌──────────────────────────────────────────────────┐
│              LangGraph StateGraph                │
│                                                  │
│  [fetch_node] ──► [cluster_node]                 │
│                        │                         │
│                        ▼                         │
│               [rewrite_node]  ← Query Rewrite    │
│         （基于对话历史改写用户指令）                │
│                        │                         │
│                        ▼                         │
│               [plan_node] ◄──────────────────┐   │
│                    │                          │   │
│          ┌─────────┴────────┐                 │   │
│          ▼                  ▼                 │   │
│      [act_node]    [summarize_node]           │   │
│       (Tools)           │                     │   │
│          └──────────────┼─────────────────────┘   │
│                         ▼                         │
│                  [reflect_node]  ← 自反思          │
│                         │                         │
│                    （结构化输出）                   │
└──────────────────────────────────────────────────┘
    │
    ▼
AgentOutput（待办 / 会议 / 跟进 / 草拟回复 / VIP摘要）
```

---

## 二·五、功能版图

按**感知 → 理解 → 行动**三个维度规划 Agent 的能力边界：

### 感知层：主动信息聚合

| 功能 | 说明 | 触发方式 |
|---|---|---|
| **早间邮件速览** | 按紧急度排序，每封一句话概要，突出今日待处理事项 | `--mode morning-briefing` |
| **按项目/人分组摘要** | 识别邮件所属项目/合同/联系人，结构化分组呈现 | 自动 |
| **VIP 发件人监控** | 可配置白名单，有新邮件时单独标出 | 自动 |
| **未回复追踪** | 扫描我方发出超 N 天未获回复的邮件，提醒跟进 | `--mode follow-up-check` |
| **对方承诺事项追踪** | 提取邮件中"I will send by Friday"类承诺句，记录截止日期 | 自动 |

### 理解层：深度语义分析

| 功能 | 说明 | 触发方式 |
|---|---|---|
| **会议信息提取** | 识别时间、地点、议程，生成结构化 `MeetingItem` | 自动 |
| **紧急度/情绪感知** | 检测措辞紧张程度，在速览中标注紧急标记 | 自动 |
| **多语言摘要** | 收到外文邮件时自动附加中文摘要 | 自动 |
| **线程对话还原** | 将多人往来邮件整理为时序对话流（基于 RAG 父链展开） | 自动 |

### 行动层：代理执行

| 功能 | 说明 | 安全机制 |
|---|---|---|
| **草拟回复** | 根据邮件内容起草回复，人工确认后发送 | `interrupt_before=["send_node"]` |
| **语气调整** | 将草稿调整为正式/轻松/简洁风格 | — |
| **发起新邮件** | "帮我给张教授写封感谢邮件" | 人工确认 |
| **标记 / 归档** | 按规则或 LLM 判断打标签、移入归档文件夹 | — |
| **创建待办** | 将需处理事项写入本地待办列表 | — |
| **周报/月报生成** | 基于 RAG 向量库回顾历史邮件，生成汇报草稿 | `--mode weekly-report` |

---

## 三、改造阶段规划

### Phase 1：基础设施升级（1～2天）

**目标**：引入 LangChain/LangGraph 依赖，打通最小可用路径，不破坏现有功能。

#### 1.1 新增依赖

```toml
# pyproject.toml
[project.dependencies]
langchain = ">=0.3"
langchain-openai = ">=0.2"      # 兼容 DeepSeek OpenAI 格式
langgraph = ">=0.2"
faiss-cpu = ">=1.8"             # 向量检索
```

#### 1.2 新增文件结构

```
src/
├── agent/
│   ├── __init__.py
│   ├── graph.py          # LangGraph StateGraph 主图定义
│   ├── nodes.py          # 各节点函数
│   ├── tools.py          # Tool 定义（fetch / reply / flag 等）
│   ├── state.py          # AgentState TypedDict
│   └── structured.py     # Pydantic 结构化输出模型
├── embeddings.py         # Embedding 聚类替换 TF-IDF
└── ...（现有文件保持不变）
```

#### 1.3 改造 `config.py`

新增字段以支持 LangChain 模型配置：

```python
@dataclass
class Config:
    ...
    # 新增
    embedding_model: str = "text-embedding-3-small"
    agent_max_iterations: int = 10
    enable_agent_mode: bool = False   # 兼容开关，False 时走旧流水线
```

---

### Phase 2：结构化输出（2天）

**目标**：用 Pydantic + `with_structured_output` 替换现有 Markdown 字符串输出。

#### 2.1 定义输出模型（`src/agent/structured.py`）

```python
from pydantic import BaseModel
from typing import List, Literal, Optional

class TodoItem(BaseModel):
    title: str                              # 待办事项标题
    priority: Literal["P0", "P1", "P2"]    # 优先级
    deadline: Optional[str] = None          # 截止时间（ISO 格式）
    related_uids: List[int] = []            # 关联邮件 UID
    action_required: bool = False           # 是否需要 Agent 采取行动
    suggested_action: Optional[str] = None  # 建议动作描述

class MeetingItem(BaseModel):
    subject: str                            # 会议主题
    time: Optional[str] = None             # 时间（ISO 格式）
    location: Optional[str] = None         # 地点或会议链接
    participants: List[str] = []           # 参与者列表
    related_uid: Optional[int] = None      # 来源邮件 UID

class FollowUpItem(BaseModel):
    description: str                        # 跟进事项描述
    direction: Literal["sent", "received"]  # sent=我方需跟进, received=对方承诺事项
    due_date: Optional[str] = None         # 预计截止时间
    related_uid: Optional[int] = None

class DraftReply(BaseModel):
    to_uid: int                             # 待回复的邮件 UID
    draft: str                              # 草拟的回复正文
    tone: Literal["formal", "casual", "concise"] = "formal"
    pending_confirmation: bool = True       # 始终需要人工确认后才发送

class AgentOutput(BaseModel):
    todos: List[TodoItem]                   # 待办事项
    meetings: List[MeetingItem] = []        # 提取的会议信息
    follow_ups: List[FollowUpItem] = []     # 需跟进事项（我方/对方承诺）
    draft_replies: List[DraftReply] = []    # 待确认的草拟回复
    vip_alerts: List[str] = []             # VIP 发件人重要邮件摘要
    summary: str                            # 整体摘要（≤200字）
    unresolved: List[str] = []             # 待确认事项
```

#### 2.2 改造 `llm_summarizer.py`

```python
# 旧实现
resp = client.chat.completions.create(model=..., messages=..., temperature=0.2)
content = resp.choices[0].message.content

# 新实现（使用 LangChain）
from langchain_openai import ChatOpenAI
from .agent.structured import AgentOutput

llm = ChatOpenAI(model=config.deepseek_model, base_url=config.deepseek_base_url, ...)
structured_llm = llm.with_structured_output(AgentOutput)
result: AgentOutput = structured_llm.invoke(messages)
```

---

### Phase 3：Embedding 聚类（2～3天）

**目标**：用语义 Embedding 替换 TF-IDF，提升跨语言、同义词场景的聚类质量。

#### 3.1 新建 `src/embeddings.py`

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def embed_and_cluster(items: List[EmailItem], config: Config) -> ClusterResult:
    embeddings = OpenAIEmbeddings(
        model=config.embedding_model,
        openai_api_key=config.deepseek_api_key,
        openai_api_base=config.deepseek_base_url,
    )
    texts = [f"{it.subject}\n{it.text[:500]}" for it in items]
    vectors = embeddings.embed_documents(texts)
    
    # 层次聚类（不需要预设 k）
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.4,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(vectors)
    # 按 label 分组 → 生成 EmailThread 列表
    ...
```

#### 3.2 兼容策略

在 `clusterer.py` 入口处加兼容判断：

```python
def cluster_emails(items, config):
    if config.enable_agent_mode:
        from .embeddings import embed_and_cluster
        return embed_and_cluster(items, config)
    # 原有 TF-IDF 路径
    return _legacy_cluster(items, config)
```

---

### Phase 4：LangGraph Agent 主图（3～5天）

**目标**：构建可动态决策的 Agent 图，支持多步推理与工具调用。

#### 4.1 定义 Agent 状态（`src/agent/state.py`）

```python
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from ..models import EmailItem, EmailThread
from .structured import AgentOutput

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    email_items: List[EmailItem]
    threads: List[EmailThread]
    output: AgentOutput | None
    actions_taken: List[str]
    instruction: str
```

#### 4.2 定义 Tools（`src/agent/tools.py`）

```python
from langchain_core.tools import tool

# ── 读取类（无副作用）──────────────────────────────────────
@tool
def fetch_email_detail(uid: int) -> str:
    """获取指定 UID 邮件的完整正文"""
    ...

@tool
def search_emails_by_topic(query: str) -> str:
    """在历史 + 本次邮件中语义搜索相关内容，返回摘要列表"""
    ...

@tool
def check_pending_replies(days: int = 3) -> str:
    """检查我方发出但超过 days 天未收到回复的邮件，返回待跟进列表"""
    ...

@tool
def extract_meeting_info(uid: int) -> str:
    """从指定邮件中提取会议时间、地点、参与者等结构化信息"""
    ...

# ── 生成类（输出草稿，不直接发送）────────────────────────────
@tool
def draft_reply(uid: int, instruction: str) -> str:
    """根据邮件内容和 instruction 指令起草回复正文，不自动发送，需人工确认"""
    ...

@tool
def draft_new_email(to: str, subject: str, instruction: str) -> str:
    """起草一封全新邮件（非回复）。to: 收件人，instruction: 写作要求，需人工确认后发送"""
    ...

@tool
def adjust_tone(draft: str, tone: str) -> str:
    """调整邮件草稿的语气。tone 可取 'formal'（正式）/ 'casual'（轻松）/ 'concise'（简洁）"""
    ...

@tool
def generate_report(period: str) -> str:
    """基于 RAG 历史邮件向量库生成汇总报告。period: 'daily' / 'weekly' / 'monthly'"""
    ...

# ── 写操作（有副作用，均需 interrupt_before=["act"] 人工确认）───
@tool
def send_email(uid: int | None, to: str, subject: str, body: str) -> str:
    """发送邮件。uid 非空时为回复，为 None 时为新邮件。需人工确认后执行。"""
    ...

@tool
def flag_email_as_important(uid: int) -> str:
    """将指定邮件标记为重要/星标"""
    ...

@tool
def archive_email(uid: int, folder: str = "Archive") -> str:
    """将指定邮件移动到归档文件夹"""
    ...

@tool
def create_todo(title: str, priority: str, deadline: str | None) -> str:
    """创建一条待办事项到本地待办列表"""
    ...
```

#### 4.3 定义图节点（`src/agent/nodes.py`）

```python
def fetch_node(state: AgentState) -> AgentState:
    """节点1：从 IMAP 抓取邮件（可跳过，若已有 email_items）"""

def cluster_node(state: AgentState) -> AgentState:
    """节点2：Embedding 聚类，生成 threads"""

def plan_node(state: AgentState) -> AgentState:
    """节点3：LLM 分析线程，规划下一步动作（调用工具 or 直接总结）"""

def act_node(state: AgentState) -> AgentState:
    """节点4：执行工具调用（ReAct 循环）"""

def summarize_node(state: AgentState) -> AgentState:
    """节点5：生成最终结构化输出"""

def router(state: AgentState) -> str:
    """条件路由：判断是否继续 act 还是进入 summarize"""
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "act"
    return "summarize"
```

#### 4.4 组装图（`src/agent/graph.py`）

```python
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import fetch_node, cluster_node, plan_node, act_node, summarize_node, router

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    
    builder.add_node("fetch",     fetch_node)
    builder.add_node("cluster",   cluster_node)
    builder.add_node("plan",      plan_node)
    builder.add_node("act",       act_node)
    builder.add_node("summarize", summarize_node)
    
    builder.set_entry_point("fetch")
    builder.add_edge("fetch",   "cluster")
    builder.add_edge("cluster", "plan")
    builder.add_conditional_edges("plan", router, {
        "act":       "act",
        "summarize": "summarize",
    })
    builder.add_edge("act", "plan")   # ReAct 循环
    builder.add_edge("summarize", END)
    
    return builder.compile()
```

---

### Phase 5：对话记忆与交互模式（2天）

**目标**：解决「用户追问」的多轮对话问题——同一 session 内用户可基于上一轮结果继续追问，LLM 能感知完整对话历史，无需重新抓取邮件。

> **适用场景**：用户先问"总结今天邮件"，再追问"其中哪些要今天回复"——两轮间的上下文由 Checkpoint 自动衔接。

#### 5.1 持久化 Checkpointer

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("agent_memory.db")
graph = build_graph().compile(checkpointer=checkpointer)

# 同一 thread_id 下自动保留上下文
config = {"configurable": {"thread_id": "session-001"}}
graph.invoke({"instruction": "总结今天的邮件"}, config=config)
# 追问：无需重新抓取
graph.invoke({"instruction": "其中哪些需要今天内回复？"}, config=config)
```

#### 5.2 改造 CLI（`src/cli.py`）

新增 `--agent` 标志与 `--mode` 参数，旧路径完全不变（向后兼容）：

```
# 交互式问答（默认 agent 模式）
email-agent --since 7d --instruction "生成待办" --agent

# 早间速览（适合每天早上 7 点 cron/Task Scheduler 触发）
email-agent --since 1d --mode morning-briefing --agent

# 未回复追踪
email-agent --mode follow-up-check --days 3 --agent

# 周报生成
email-agent --mode weekly-report --agent

# 旧路径，不变
email-agent --since 7d --instruction "生成待办"
```

`--mode` 本质上是预设的 `instruction` 快捷方式，在 CLI 层展开为具体指令字符串后送入 Agent 图，无需修改图结构。

> **参数说明**：`--days N` 仅在 `--mode follow-up-check` 时生效，表示"超过 N 天未回复则视为待跟进"，默认值为 3。

---

### Phase 6：RAG 历史邮件向量库（3～4天）

**目标**：解决「跨天邮件线程上下文丢失」问题——当今天收到某封邮件的后续回复时，能自动检索并注入历史相关邮件作为上下文，而不依赖 UID 增量过滤。

> **问题根源**：现有 `last_seen_uid` 过滤机制会屏蔽所有旧 UID 邮件，导致 A 昨天的原始邮件在今天处理 A 的后续回复时完全不可见。LangGraph Checkpoint 无法解决此问题（它只保留对话消息，不存储邮件原文）。

#### 6.1 检索策略：三级流水线

邮件场景的最优检索策略是**先确定性、再统计性、最后语义精排**，分三步完成：

```
新邮件到达
    │
    ▼
Step 0：解析 In-Reply-To / References 邮件头（确定性，无检索，零成本）
    ├─ 有父 message_id → 按 message_id 精确拉取父链邮件 ✓
    │   （覆盖绝大多数正常回复场景，100% 准确）
    │
    └─ 无 / 父链不完整（转发、外部回复等边缘情况）
         │
         ▼
Step 1：metadata 预过滤 + Hybrid 粗召回（Top-20）
    - 先按「发件人域名 + 日期窗口」过滤，缩小候选集
    - BM25（精确词匹配：人名、项目代号、专有名词）
    - FAISS 向量检索（语义相似：同义词、跨语言）
    - RRF 融合两路结果（只看排名，不受分数量纲影响）
         │
         ▼
Step 2：Cross-encoder Rerank（细召回 Top-3）
    - 用 BAAI/bge-reranker-v2-m3（本地推理，免费）
    - 对粗召回的 20 条结果做精排，取 Top-3 注入 prompt
```

> **为什么不直接用纯向量检索**：BM25 对人名、项目编号、邮件主题中的专有名词 recall 更好；向量检索覆盖语义相近但措辞不同的情况。两路互补，RRF 融合比线性加权更鲁棒（无需调分数权重）。

#### 6.2 文档构建策略：剥离引用 + metadata 保留线程关系

**核心设计原则**：`page_content` 只存干净的新鲜正文（负责 embedding 质量），线程父子关系存进 `metadata`（负责检索后的上下文还原）。两件事分开，互不干扰。

> **为什么不在 page_content 里保留引用**：
> - 邮件回复通常会 quote 历史内容，一封"好的，我同意"的短回复可能 90% 内容都是引用，embedding 向量会偏向历史内容而非这封邮件本身
> - BM25 的词频统计会被引用内容污染
> - 父邮件已单独入库，引用内容完全重复
>
> **线程关系如何还原**：检索到子邮件后，通过 metadata 中的 `in_reply_to` 字段精确拉取父邮件，在注入 prompt 前做「父链展开」，LLM 仍能看到完整对话。

首先需要在 `EmailItem` 模型中新增线程头字段（`src/models.py`）：

```python
@dataclass
class EmailItem:
    uid: int
    date: datetime
    from_addr: str
    subject: str
    text: str
    mailbox: str
    message_id: Optional[str] = None
    in_reply_to: Optional[str] = None    # 新增：直接父邮件 message_id
    references: Optional[str] = None     # 新增：完整父链 message_id 列表（空格分隔）
    to_addrs: List[str] = field(default_factory=list)
    cc_addrs: List[str] = field(default_factory=list)
    historical_context: str = ""         # 新增：RAG 检索后注入的历史上下文
```

构建历史邮件向量库（`src/email_store.py`）：

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
import re

def strip_quoted_content(text: str) -> str:
    """剥离邮件中的引用历史内容，只保留新鲜正文"""
    lines = text.splitlines()
    fresh_lines = []
    for line in lines:
        stripped = line.strip()
        # 剥离 > 引用行
        if stripped.startswith(">"):
            continue
        # 剥离 "On ... wrote:" 分隔符及其后内容
        if re.match(r"^On .+wrote:$", stripped):
            break
        fresh_lines.append(line)
    return "\n".join(fresh_lines).strip()

def build_email_document(item: EmailItem) -> Document:
    """
    构建入库 Document：
    - page_content = 剥离引用后的干净正文（负责 embedding/BM25 质量）
    - metadata     = 完整元信息 + 线程父子关系（负责检索后上下文还原）
    """
    fresh_text = strip_quoted_content(item.text or "")
    # 若剥完引用后内容为空（纯转发），退回使用原始文本
    body = fresh_text[:1000] if fresh_text else (item.text or "")[:1000]

    return Document(
        page_content=body,          # 干净正文，subject 不重复写入
        metadata={
            "uid": item.uid,
            "subject": item.subject,           # subject 进 metadata
            "from": item.from_addr,
            "message_id": item.message_id or "",
            "in_reply_to": item.in_reply_to or "",   # 直接父邮件 ID
            "references": item.references or "",      # 完整父链 ID 列表
            "date": item.date.isoformat(),
            "mailbox": item.mailbox,
        },
    )

def build_or_update_store(items: List[EmailItem], config: Config) -> FAISS:
    """将处理过的邮件增量写入 FAISS 向量库，持久化到本地。
    若本地已有向量库则追加（update），否则全量初始化（build）。
    """
    embeddings = OpenAIEmbeddings(
        model=config.embedding_model,
        openai_api_key=config.deepseek_api_key,
        openai_api_base=config.deepseek_base_url,
    )
    docs = [build_email_document(it) for it in items]
    store_path = config.email_store_path

    if os.path.exists(store_path):
        # 增量追加：加载现有向量库后 add_documents，不重建
        store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        store.add_documents(docs)
    else:
        # 首次全量初始化
        store = FAISS.from_documents(docs, embeddings)

    store.save_local(store_path)
    return store

def load_store(config: Config) -> FAISS | None:
    """从本地加载 FAISS 向量库；若不存在（首次运行）返回 None。"""
    store_path = config.email_store_path
    if not os.path.exists(store_path):
        return None
    embeddings = OpenAIEmbeddings(
        model=config.embedding_model,
        openai_api_key=config.deepseek_api_key,
        openai_api_base=config.deepseek_base_url,
    )
    return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)

def expand_with_parents(
    retrieved_docs: List[Document],
    msg_id_index: dict,
    max_depth: int = 3,
) -> List[Document]:
    """
    检索后展开父链：
    对每条检索结果，若其 in_reply_to 指向历史邮件，则递归拉取父邮件追加进上下文。
    最终 LLM 能看到完整对话链，而不只是一封孤立的回复。
    """
    expanded = list(retrieved_docs)
    seen = {d.metadata.get("message_id") for d in expanded}
    queue = list(retrieved_docs)
    depth = 0

    while queue and depth < max_depth:
        next_queue = []
        for doc in queue:
            parent_id = doc.metadata.get("in_reply_to", "")
            if parent_id and parent_id not in seen and parent_id in msg_id_index:
                parent_doc = msg_id_index[parent_id]
                expanded.append(parent_doc)
                seen.add(parent_id)
                next_queue.append(parent_doc)
        queue = next_queue
        depth += 1

    # 按日期排序，让 LLM 看到时间顺序的对话链
    expanded.sort(key=lambda d: d.metadata.get("date", ""))
    return expanded

def build_hybrid_retriever(docs: List[Document], vectorstore: FAISS) -> ContextualCompressionRetriever:
    """构建 BM25 + FAISS 混合检索 + Cross-encoder Rerank 的三级检索器"""
    # 粗召回：BM25 + 向量，RRF 融合
    bm25 = BM25Retriever.from_documents(docs, k=20)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    ensemble = EnsembleRetriever(
        retrievers=[bm25, faiss_retriever],
        weights=[0.4, 0.6],   # 邮件场景向量权重稍高
    )
    # 细召回：Cross-encoder Rerank
    reranker = CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3"),
        top_n=3,
    )
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble,
    )
```

#### 6.3 在 fetch_node 中实现三级检索 + 父链展开（`src/agent/nodes.py`）

```python
def fetch_node(state: AgentState) -> AgentState:
    # Step A：正常增量抓取新邮件
    new_items = fetcher.fetch_unread(...)
    store = load_store(config)

    if store:
        all_docs = list(store.docstore._dict.values())
        retriever = build_hybrid_retriever(all_docs, store)
        # 建立 message_id → doc 的索引，用于精确父链查找
        msg_id_index = {
            d.metadata["message_id"]: d
            for d in all_docs
            if d.metadata.get("message_id")
        }

        for item in new_items:
            # Step 0：In-Reply-To 确定性查找（优先，零成本）
            ref_ids = (item.references or "").split()   # References 头包含完整父链
            parent_docs = [
                msg_id_index[rid] for rid in ref_ids if rid in msg_id_index
            ]

            if parent_docs:
                # 按日期排序后直接注入，100% 准确，无需检索
                parent_docs.sort(key=lambda d: d.metadata.get("date", ""))
                item.historical_context = "\n---\n".join(
                    d.page_content for d in parent_docs
                )
            else:
                # Step 1 + 2：Hybrid 粗召回 + Rerank（兜底）
                query = f"{item.from_addr} {item.subject}"
                retrieved = retriever.invoke(query)   # BM25+FAISS → Rerank → Top-3

                # 父链展开：对检索结果递归追加父邮件，还原完整对话链
                expanded = expand_with_parents(retrieved, msg_id_index, max_depth=3)
                item.historical_context = "\n---\n".join(
                    d.page_content for d in expanded
                )

    # 处理完成后将本批邮件写入向量库（供未来检索）
    build_or_update_store(new_items, config)
    return {**state, "email_items": new_items}
```

**数据流说明**：

```
检索到子邮件 B（干净正文，无引用噪音）
    │
    ├─ Step 0 命中：References 头包含父链 ID → 直接精确拉取父邮件 A
    │
    └─ Step 0 未命中：Hybrid 检索召回相关邮件
         │
         └─► expand_with_parents() 递归追加父链
              │
              └─► 注入 prompt：[A 的内容（日期排序）] + ... + [B 的内容]
                   → LLM 看到完整时序对话，而 embedding 质量未受引用污染
```

#### 6.4 新增依赖与 Config 字段

```toml
# pyproject.toml 新增
rank-bm25 = ">=0.2"          # BM25 实现
sentence-transformers = ">=3" # Cross-encoder Rerank 本地模型
```

```python
@dataclass
class Config:
    ...
    email_store_path: str = "email_vector_store"   # FAISS 本地存储路径
    rag_bm25_weight: float = 0.4                   # BM25 在 RRF 中的权重
    rag_rerank_top_n: int = 3                      # Rerank 后保留条数
    rag_rerank_model: str = "BAAI/bge-reranker-v2-m3"
```

#### 6.5 两种 Memory 机制的职责划分

| 机制 | 解决的问题 | 存储内容 | 检索方式 |
|---|---|---|---|
| **LangGraph Checkpoint**（Phase 5） | 用户多轮追问的对话连贯性 | 完整消息历史（HumanMsg / AIMsg） | 按 `thread_id` 精确读取，无检索 |
| **RAG 历史邮件向量库**（Phase 6） | 跨天邮件线程上下文丢失 | 历史邮件原文的语义向量 + BM25 索引 | Step 0 精确 → Step 1 Hybrid → Step 2 Rerank |

两者互补，各司其职，同时使用才能完整覆盖项目的记忆需求。

---

## 四、各阶段工作量估计

| 阶段 | 内容 | 估计工时 | 风险 |
|---|---|---|---|
| Phase 1 | 依赖引入 + 文件结构 | 0.5天 | 低 |
| Phase 2 | 结构化输出 | 1天 | 低 |
| Phase 3 | Embedding 聚类 | 2天 | 中（需 API 费用） |
| Phase 4 | LangGraph 主图 | 4天 | 高（设计复杂） |
| Phase 5 | Checkpoint 对话记忆（解决多轮追问） | 1天 | 低 |
| Phase 6 | RAG 历史邮件向量库（解决跨天线程上下文丢失） | 3天 | 中 |
| **合计** | | **约 11.5天** | |

---

## 五、改造前后对比

| 维度 | 改造前 | 改造后 |
|---|---|---|
| 架构模式 | 固定线性流水线 | LangGraph 动态图 |
| LLM 角色 | 被动总结器 | 主动推理 + 工具调用 |
| 聚类算法 | TF-IDF + cosine | OpenAI Embedding + 层次聚类 |
| 输出格式 | 原始 Markdown 字符串 | Pydantic 结构化对象 |
| 交互模式 | 单次批处理 | 多轮对话（带 Memory）/ 定时模式（morning-briefing 等） |
| 动作能力 | 无 | 回复/发起邮件 / 语气调整 / 标记 / 归档 / 创建待办 / 提取会议 / 跟进追踪 / 生成报告 |
| 可扩展性 | 硬编码逻辑 | 注册新 Tool 即可扩展 |
| 向后兼容 | — | `--agent` 开关控制，旧路径保留 |

---

## 六、推荐实施顺序

```
Phase 2（结构化输出）
    ↓  最快见效，改动最小
Phase 1（依赖 + 文件结构）
    ↓  为后续打基础
Phase 4（LangGraph 主图）
    ↓  核心改造，分多个 PR 完成
Phase 3（Embedding 聚类）
    ↓  可并行或在 Phase 4 完成后进行
Phase 5（Checkpoint 对话记忆）
    ↓  解决多轮追问，改动较小
Phase 6（RAG 历史邮件向量库）
    ↓  解决跨天线程上下文丢失，收尾优化
```

> **建议**：先单独完成 Phase 2，用最小代价验证结构化输出的价值，再推进 Phase 4 的图架构改造。Phase 5 与 Phase 6 解决不同层次的记忆问题，优先做 Phase 5（改动小），再做 Phase 6（补充 RAG）。

---

## 七、注意事项

1. **API 费用**：Embedding 模型按 token 计费，建议开发阶段用本地模型（如 `sentence-transformers`）替代，生产再换 OpenAI。
2. **Tool 安全**：`reply_to_email` 等写操作 Tool 建议增加**人工确认**步骤（`interrupt_before=["act"]`），防止误操作。
3. **兼容性**：全程保留旧流水线，通过 `enable_agent_mode` / `--agent` 开关控制，不影响现有用户。
4. **测试**：LangGraph 图的每个节点应单独编写单元测试，避免端到端测试依赖外部 API。
5. **两种 Memory 机制不可混淆**：
   - **LangGraph Checkpoint**（Phase 5）= 短期对话记忆，解决同一 session 内的多轮追问问题，存的是消息历史，按 `thread_id` 精确读取，无需检索。
   - **RAG 历史邮件向量库**（Phase 6）= 长期知识库，解决跨天/跨 session 的邮件线程上下文丢失问题，存的是邮件语义向量，通过相似度检索召回。
   - 两者互补，不能互相替代。

---

## 七、已实现的迭代增强

### Phase 7a：交互式多轮对话（Chat Loop）

**背景**：原有 CLI 每次运行后即退出，用户追问需要重新启动进程，不便于连续对话。

**实现方案**：新增 `--chat` 参数，程序保持运行，持续读取用户输入，直到输入 `exit` 退出。

```bash
python -m src.cli --since 7d --include-seen --chat --session my-session
```

**关键设计**：
- 第一轮：传完整 `initial_state`（含空 `email_items`），触发 `fetch_node` 拉取邮件
- 后续轮：只传 `instruction` 等必要字段，**不传 `messages`**，让 LangGraph 从 SQLite Checkpoint 自动恢复完整对话历史
- `fetch_node` 检测到 `email_items` 非空（由 checkpoint 恢复）时自动跳过重新拉取
- `Ctrl+C` / `EOF` 安全退出

```
============================================================
[Chat] 交互式对话模式启动（session=my-session）
[Chat] 邮件将在第一次指令时拉取，后续对话复用同一批邮件
[Chat] 输入 exit / quit / q 退出
============================================================

[Agent] 处理中：根据邮件生成我的待办...
...输出报告...

------------------------------------------------------------
>>> 把 P0 的任务详细说说
[QueryRewrite] 原始: 把 P0 的任务详细说说
[QueryRewrite] 改写: COMP 1110 作业（P0优先级）的具体要求和截止时间是什么？
...
```

**涉及文件**：`src/cli.py`（新增 `_run_chat_mode` 函数，`--chat` 参数）

---

### Phase 7b：Query Rewrite（基于对话历史的查询改写）

**背景**：多轮对话中用户常使用代词或省略语（"这封"、"他"、"上面那个"），若直接传入 `plan_node`，LLM 缺乏上下文会误判或无法检索。

**实现方案**：在 `cluster_node` 和 `plan_node` 之间插入 `rewrite_node`，利用对话历史将模糊指令改写为自包含的精确查询。

**图结构变化**：
```
旧：fetch → cluster → plan
新：fetch → cluster → rewrite → plan
```

**`rewrite_node` 行为**：

| 情况 | 行为 | LLM 调用 |
|------|------|----------|
| 首轮（无历史 messages）| 直接透传原始 instruction | ❌ 不调用 |
| 后续轮（有历史）| 提取最近 6 条历史 → LLM 改写为自包含查询 | ✅ 1次 |

**改写示例**：
```
历史：[Human] 根据邮件生成待办
      [AI] 发现 COMP 1110 作业在 3月5日 截止...
原始指令：这个作业什么时候上交？
改写后：COMP 1110 作业的截止日期是何时？
```

**改写规则**（system prompt）：
1. 补全代词（"他"/"这封"/"那个"）为具体实体
2. 补全省略的时间/对象/条件
3. 保留用户原始意图，不添加不存在的假设
4. 若原始指令已足够清晰，原样返回
5. 只输出改写后文本，不附带任何解释

**`AgentState` 新增字段**：
```python
rewritten_instruction: str  # rewrite_node 改写后的指令；首轮与 instruction 相同
```

`plan_node` 优先读取 `rewritten_instruction`，使 RAG 检索和工具调用更加精准。

**涉及文件**：
- `src/agent/state.py`（新增 `rewritten_instruction` 字段）
- `src/agent/nodes.py`（新增 `rewrite_node`，`plan_node` 读取 `rewritten_instruction`）
- `src/agent/graph.py`（注册 `rewrite` 节点，更新边 `cluster→rewrite→plan`）
- `src/cli.py`（所有 `initial_state` 初始化新增 `rewritten_instruction: ""`）

---

## 八、未来考虑项（当前阶段不实现）

以下两项是架构成熟后可叠加的优化方向，不影响现有 6 个 Phase 的实施。

---

### F1：Reflection（自我反思）

**解决的问题**：基础 ReAct 循环中，Agent 生成输出后直接结束，不会评估自己是否遗漏了关键邮件、草拟的回复语气是否合适。Reflection 在输出后增加一轮自评估，发现问题可回头修正。

**图结构变化**：在 `summarize_node` 之后新增 `reflect_node`，形成可选的修正循环：

```
summarize_node（生成初稿输出）
    │
    ▼
reflect_node（LLM 自评：是否有遗漏？草稿质量是否达标？）
    │
    ├─ critique 不为空 → 带 critique 消息回到 plan_node 重新推理
    │
    └─ 质量满足（或达到最大反思次数）→ END
```

**实现要点**：

```python
def reflect_node(state: AgentState) -> AgentState:
    """让 LLM 对当前输出做自评估，返回 critique 或空字符串"""
    critique_prompt = f"""
你刚刚生成了以下邮件处理结果：
{state['output'].model_dump_json(indent=2)}

请检查：
1. 是否有紧急邮件未被纳入 P0 待办？
2. 草拟回复的语气是否符合上下文？
3. 是否遗漏了需要跟进的承诺事项？

如果发现问题，请简述 critique；如果质量满足，直接回复"OK"。
"""
    ...

def reflection_router(state: AgentState) -> str:
    """根据 reflect_node 的 critique 决定是否重新推理"""
    critique = state.get("critique", "")
    if critique and critique.strip() != "OK" and state["reflection_count"] < 2:
        return "plan"   # 带 critique 重新推理
    return END
```

**适用场景**：对草拟回复、周报生成等精度要求较高的输出启用；日常速览不需要。可通过 `AgentState` 中的 `enable_reflection: bool` 字段按需开关。

**代价**：每次输出多一次 LLM 调用；最多反思 2 次（防止无限循环）。

> **⚠️ 实现时需同步扩展 `AgentState`**（Phase 4.1）：
> ```python
> class AgentState(TypedDict):
>     ...                                    # 原有字段不变
>     critique: str                          # 新增：reflect_node 的批评内容，空字符串表示通过
>     reflection_count: int                  # 新增：已反思次数，上限为 2 次
>     enable_reflection: bool                # 新增：是否开启 Reflection（按场景开关）
> ```

---

### F2：RAG Context Compression（上下文压缩）

**解决的问题**：Phase 6 的父链展开可能引入过多历史邮件内容，导致注入 prompt 的 `historical_context` 过长，超出 context window 或增加无关噪音。Context Compression 在注入前对检索结果做精简，只保留与当前新邮件真正相关的句子。

**为什么现阶段不需要**：
- 当前设计中 `page_content` 已剥离引用，每封邮件截断至 1000 字符，父链展开上限 `max_depth=3`，实际注入量可控
- 邮件正文通常比文档/网页短得多，压缩收益有限
- 引入压缩器会增加一次额外 LLM/模型调用，得不偿失

**未来触发条件**：当单次 `historical_context` 超过某个 token 阈值（如 3000 tokens）时，启用压缩。

**实现方案**：使用 LangChain 内置的 `LLMChainFilter` 或 `EmbeddingsFilter`：

```python
from langchain.retrievers.document_compressors import LLMChainFilter, EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# 方案 A：LLM 过滤（精度高，多一次 LLM 调用）
compressor = LLMChainFilter.from_llm(llm)

# 方案 B：Embedding 相似度过滤（速度快，无额外 LLM 调用，推荐）
compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever,   # 接在现有三级检索之后
)
```

**与现有架构的关系**：直接在 `build_hybrid_retriever()` 末尾追加压缩层，其余代码无需改动。推荐使用方案 B（`EmbeddingsFilter`），与现有 Embedding 模型复用，零额外 API 费用。

---

### F3：MCP 化（工具服务化，跨客户端复用）

**解决的问题**：当前 `@tool` 方式的工具是**进程内函数调用**，只能被同一 Python 进程内的 LangGraph Agent 使用。若未来希望从 Claude Desktop、Cursor、其他 Agent（如"办公室 Agent"）直接调用邮件工具，需要将工具改造为 **MCP Server**——通过标准 JSON-RPC 协议对外暴露，实现语言无关、进程隔离的工具服务。

> **与直接 import tools.py 的区别**：
> - `@tool` 方式：对方必须是 Python 项目，必须安装你的所有依赖（imapclient、faiss 等），必须有你的 `.env` 配置，代码耦合度高
> - MCP 方式：对方只需启动你的进程或连接你的地址，发送/接收 JSON，无需知道你用什么语言、什么库、什么配置

**适用场景**：
- 想让 Claude Desktop / Cursor 等客户端直接调用邮件工具
- 未来要把邮件工具集成进其他 Agent 项目（如办公室综合 Agent），且对方不一定是 Python
- 工具数量继续增长，需要独立部署和更新，不希望重启主 Agent

**建议的 MCP Server 分组**：

| MCP Server | 包含工具 | 说明 |
|---|---|---|
| `email_server` | `fetch_email_detail` / `send_email` / `flag_email_as_important` / `archive_email` | 依赖 IMAP，独立进程隔离连接 |
| `knowledge_server` | `search_emails_by_topic` / `check_pending_replies` | 依赖 FAISS/BM25，独立进程隔离向量库 |
| `compose_server` | `draft_reply` / `draft_new_email` / `adjust_tone` / `generate_report` | 纯 LLM 生成，无 IMAP 依赖 |

轻量工具（`create_todo` / `extract_meeting_info`）可暂时保留为 `@tool`，不需要全部 MCP 化。

**LangGraph 侧接入方式**（图结构不变）：

```python
from langchain_mcp_adapters import MCPToolkit

# 把 MCP Server 的工具自动转成 LangChain @tool，Agent 图无需修改
toolkit = MCPToolkit(server_params=[
    {"command": "python", "args": ["src/mcp_servers/email_server.py"]},
    {"command": "python", "args": ["src/mcp_servers/knowledge_server.py"]},
    {"command": "python", "args": ["src/mcp_servers/compose_server.py"]},
])
tools = await toolkit.get_tools()
```

**新增文件结构**：

```
src/
├── mcp_servers/
│   ├── email_server.py      # IMAP 操作工具（fetch / send / flag / archive）
│   ├── knowledge_server.py  # RAG 检索工具（search / check_pending）
│   └── compose_server.py    # 草稿生成工具（draft / adjust_tone / report）
└── agent/
    └── tools.py             # 保留轻量 @tool，或全部替换为 MCP toolkit
```

**实施前提**：需先完成 Phase 4（工具逻辑已实现），MCP 化只是把现有函数包上网络协议层，逻辑本身不变。预计工作量 **1～2 天**。
