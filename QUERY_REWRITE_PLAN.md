# Query Rewrite 增强方案

> 基于现有 `rewrite_node`（上下文补全）进行扩展，分优先级逐步实现。

---

## 一、现状分析

### 当前 `rewrite_node` 能力

```
首轮 → 直接透传（不消耗 token）
后续轮 → 取最近 6 条历史 → LLM 补全代词/省略 → rewritten_instruction
```

**缺口：**
| 能力 | 现状 |
|------|------|
| 代词/省略补全 | ✅ 已实现 |
| 改写质量校验 | ❌ 无，漂移无感知 |
| 复杂查询分解 | ❌ 单一 instruction，无多路意图 |
| HyDE 语义增强 | ❌ 用问句向量搜文档向量，空间错配 |
| 意图路由（按需改写） | ❌ 所有 Query 都走 LLM，浪费 token |

---

## 二、增强功能详细设计

### P1（次优先级）— 语义漂移过滤

#### 问题

LLM 改写后没有任何质量校验。若 LLM 出现幻觉（改写后语义严重偏离），直接进 FAISS 检索，召回结果完全错误但系统无感知。

**注**：邮件指令通常包含明确实体（人名/项目名/时间），比开放域问答漂移概率更低，属于中优先级安全兜底，不是最紧迫需求。

#### 方案

在 `rewrite_node` 内部，改写完成后立即用 `all-MiniLM-L6-v2` 计算原始 instruction 与改写结果的余弦相似度：

```
原始 instruction  → Embedding → vec_original
改写后 instruction → Embedding → vec_rewritten

cosine_sim = dot(vec_original, vec_rewritten)

if cosine_sim < DRIFT_THRESHOLD (建议 0.75):
    用原始 instruction（回退）
    log WARNING: 改写漂移过大，已回退
else:
    用改写后 instruction
```

#### 实现位置

`src/agent/nodes.py` → `rewrite_node()` 尾部追加，约 10 行。

#### 成本

- 使用已有 `all-MiniLM-L6-v2`，本地 CPU，无额外 API 调用。
- 每次改写多 2 次 embed（原始 + 改写后），约 2ms，可忽略。

---

### P0（最高优先级）— 前置意图路由

#### 问题

当前所有 Query 都走 LLM 改写。简单明确的指令如 `"列出未读邮件"` 完全不需要改写，白白消耗 LLM token 和延迟。

#### 方案

在 `rewrite_node` 最开始，通过规则+轻量判断决定是否需要改写：

```
触发改写的条件（满足任一）：
  1. messages 历史 ≥ 1 条（多轮对话，可能有省略/代词）
  2. instruction 含有指代词：他/她/它/这封/那个/上一封/刚才/之前
  3. instruction 长度 < 15 字且不含实体（过短 → 可能省略了主语）
  4. instruction 含 "比较"/"vs"/"和...的区别"（多实体比较 → 需子查询分解）

不触发改写：
  → 首轮 + 指令自包含 + 无指代词 → 直接透传（当前已有首轮透传，扩展此逻辑）
```

#### 实现位置

`src/agent/nodes.py` → `rewrite_node()` 头部，`if not messages` 判断之后追加规则检查。

#### 成本

纯规则，无模型调用，O(1)。

---

### P2（低优先级）— HyDE（Hypothetical Document Embedding）

#### 问题

用户的问句（Query）与邮件正文处于不同的向量"空间"：
- 问句：`"关于Q4财务汇报的邮件"`（疑问句结构，短）
- 邮件：`"Hi, 请查收Q4财务数据汇总，附件见..."`（陈述句结构，长）

TF-IDF + FAISS 直接用问句向量匹配邮件向量，存在结构性偏差。

**注**：邮件场景已有 message_id 精确匹配（Step 0，100% 准确）和 BM25 关键词检索兜底，HyDE 主要解决纯语义问答型 RAG 中"零结构信号"的召回问题，在本项目性价比较低，建议 P0/P1 功能稳定后再评估。

#### 方案

改写阶段额外生成一封"假想邮件"，作为稠密检索的 anchor：

```
用户 Query: "帮我找Q4财务汇报相关的邮件"
                   │
                   ▼
            LLM 生成 HyDE 文档:
            "发件人: finance@company.com
             主题: Q4 财务汇报
             正文: 附上Q4各部门财务数据汇总，请查收..."
                   │
                   ▼
            用 HyDE 文档向量 → FAISS 检索（替代原始问句向量）
            用原始 Query → BM25 检索（关键词匹配不变）
                   │
                   ▼
            RRF 合并两路结果
```

#### 实现位置

**方案 A（改写层实现）**：`rewrite_node` 生成 HyDE 文档，写入新 state 字段 `hyde_document: str`，`fetch_node` 的 RAG 检索时优先用 `hyde_document` 向量而非 `instruction` 向量。

**方案 B（检索层实现）**：`build_hybrid_retriever` 增加 HyDE 检索路径，作为第三路输入 RRF。

推荐方案 B，改动更内聚，不污染 rewrite_node。

#### State 变更

```python
# state.py 新增（方案 A）
hyde_document: str        # HyDE 假想文档，用于稠密检索
```

#### 成本

- 多 1 次 LLM 调用（生成 HyDE 文档）+ 1 次 embed
- 可与现有改写合并为同一次 LLM 调用（structured output 同时输出 rewritten_instruction + hyde_document）

---

### P0（最高优先级）— 子查询分解（Sub-query Decomposition）

#### 问题

`"总结上周A项目和B项目的进展，并做比较"` 包含两个独立意图：
- `A项目进展`
- `B项目进展`

当前单一 `rewritten_instruction` 无法表达多路意图，FAISS 单次检索召回偏向其中一个。

#### 方案

对"比较类"/"多实体类"Query，分解为子查询列表，并行检索后合并：

```
原始 Query: "比较一下A项目和B项目上周的进展"
                   │
                   ▼
         LLM 分解（Structured Output）:
         sub_queries = ["A项目上周进展", "B项目上周进展"]
                   │
          ┌────────┴────────┐
          ▼                 ▼
     FAISS("A项目")    FAISS("B项目")     ← 并行检索
          │                 │
          └────────┬────────┘
                   ▼
              合并去重（union，按 uid 去重）
                   │
                   ▼
              plan_node（已有全部相关邮件）
```

#### State 变更

```python
# state.py 新增
sub_queries: List[str]    # 子查询列表，单意图时长度为 1
```

#### 实现位置

`rewrite_node` 通过 Structured Output 同时输出：
```python
class RewriteOutput(BaseModel):
    rewritten_instruction: str    # 主改写结果（给 plan_node 的 prompt）
    sub_queries: List[str]        # 给 FAISS 的多路检索 Query
    is_multi_intent: bool         # 是否需要多路检索
    hyde_document: str            # HyDE 假想文档（空字符串表示不生成）
```

一次 LLM 调用解决所有改写需求，不增加额外延迟。

---

## 三、整体架构（增强后）

```
用户 instruction
        │
        ▼
┌─────────────────────────────────┐
│         rewrite_node            │
│                                 │
│  Step 1: 意图路由（规则，O(1)）  │
│    ├─ 简单明确 → 直接透传        │
│    └─ 复杂/多轮/指代 → 进入改写  │
│                                 │
│  Step 2: LLM 改写               │
│    输出 RewriteOutput:           │
│    ├─ rewritten_instruction     │
│    ├─ sub_queries               │
│    └─ hyde_document             │
│                                 │
│  Step 3: 漂移检测（Embedding）   │
│    cosine(原始, 改写) < 0.75     │
│    → 回退到原始 instruction      │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│    fetch_node（RAG 检索增强）    │
│                                 │
│  Step 0: message_id 精确查找    │
│  Step 1: BM25(sub_queries 合并) │
│          + FAISS(hyde_document  │
│            向量 或 instruction) │
│  Step 2: RRF → Rerank → Top-K   │
└─────────────────────────────────┘
        │
        ▼
   plan_node / summarize_node
```

---

## 四、实施优先级与工作量

| 功能 | 优先级 | 工作量 | 依赖 | 价值 |
|------|--------|--------|------|------|
| 意图路由 | P0 | 小（~20行规则） | 无 | 最高性价比，节省 token，零延迟 |
| 子查询分解 | P0 | 中（~60行 + state字段 + 并行检索） | Structured Output | 多实体比较真实需求，当前完全缺失 |
| 语义漂移过滤 | P1 | 小（~15行） | 已有 all-MiniLM | 安全兜底，邮件场景漂移概率低 |
| HyDE | P2 | 中（~50行 + state字段） | Structured Output | 邮件已有精确匹配+BM25，性价比有限 |

---

## 五、不建议实现的内容

### Step-back Prompting
本项目是邮件操作 Agent，Query 已经非常具体（"找某封邮件"、"提取待办"），Step-back 抽象化后反而会损失关键信息（如人名、项目名）。**不适合本场景。**

### 小模型专项改写
本项目已使用 DeepSeek，改写 prompt 极短（< 500 token），延迟主要来自网络而非模型规模。引入另一个小模型增加系统复杂度，收益不明显。**可选，暂不优先。**

---

## 六、代码变更清单

实现 P0 后的最小变更：

```
src/agent/
  ├── state.py         + sub_queries: List[str]
  │                    + hyde_document: str
  ├── nodes.py         rewrite_node() 增加：
  │                      - 意图路由规则（头部）
  │                      - RewriteOutput structured output
  │                      - 漂移检测（尾部）
  └── structured.py    + RewriteOutput(BaseModel)

src/email_store.py     build_hybrid_retriever() 增加：
                         - hyde_document 向量作为 FAISS 查询替代
                         - sub_queries 合并 BM25 检索
```
