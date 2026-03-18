# EmailAgent 评估方案

本文档汇总 emailAgent 各组件的评估方法，覆盖 RAG 检索、Agent 轨迹、结构化输出、Query Rewrite 四个层次。

---

## 一、RAG 检索层：RAGAS

适用工具：`search_emails_by_topic`、`generate_report`（底层均走混合检索）

### 核心指标

| 指标 | 含义 | 本项目对应 |
|------|------|-----------|
| **Context Recall** | 答案所需信息是否都被召回 | 检索到的邮件是否包含用户问题的关键信息 |
| **Context Precision** | 召回内容中有多少是真正相关的 | Top-K 邮件里有多少与问题直接相关 |
| **Faithfulness** | 生成答案是否忠于检索上下文（不编造）| LLM 总结是否添加了邮件里不存在的信息 |
| **Answer Relevance** | 答案是否切题 | 输出的待办/摘要是否回答了用户指令 |

### 最小运行示例

```python
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, faithfulness, answer_relevancy
from datasets import Dataset

# 构造测试集（需人工标注 ground_truth）
data = {
    "question":        ["最近有关项目A的邮件说了什么？"],
    "contexts":        [["邮件正文1...", "邮件正文2..."]],  # 检索到的上下文
    "answer":          ["项目A需要在3月10日前提交初稿..."],  # Agent 输出
    "ground_truth":    ["项目A的截止日期是3月10日，需提交初稿"],  # 人工标注
}
dataset = Dataset.from_dict(data)

result = evaluate(
    dataset,
    metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
)
print(result)
```

### 构造 Ground Truth 的建议

- 从真实邮件中选取 10-20 个有代表性的查询场景
- 人工标注每个查询"理想上应该召回哪些邮件UID"和"正确答案是什么"
- 覆盖场景：时间敏感查询、多线程合并查询、RAG 历史上下文查询

### 三级检索的消融对比

| 检索策略 | Context Precision | Context Recall | 备注 |
|---------|-------------------|---------------|------|
| 仅 BM25 | - | - | 精确词匹配，对同义词弱 |
| 仅 FAISS | - | - | 语义召回，对关键词弱 |
| BM25 + FAISS RRF | - | - | 粗召回融合 |
| BM25 + FAISS RRF + Rerank | - | - | 启用 `RAG_RERANK_MODEL` 后 |

> 运行不同配置对比，填入实测数值，可作为论文/报告的实验结果。

---

## 二、Agent 轨迹层：Trajectory Evaluation

评估 ReAct Agent 的工具调用链路是否合理。

### 2a. 工具调用准确率（Tool F1）

构造测试用例，标注给定指令下"应该调用哪些工具"：

```python
TEST_CASES = [
    {
        "instruction": "这封邮件的截止日期是什么时候？",
        "expected_tools": ["fetch_email_detail"],  # 必须调用，否则截止日期被截断
    },
    {
        "instruction": "有没有超过3天没有回复的邮件？",
        "expected_tools": ["check_pending_replies"],
    },
    {
        "instruction": "最近关于项目A的邮件说了什么？",
        "expected_tools": ["search_emails_by_topic"],
    },
]

def evaluate_tool_calls(result_messages, expected_tools):
    actual_tools = [
        msg.tool_calls[0]["name"]
        for msg in result_messages
        if hasattr(msg, "tool_calls") and msg.tool_calls
    ]
    precision = len(set(actual_tools) & set(expected_tools)) / len(actual_tools) if actual_tools else 0
    recall    = len(set(actual_tools) & set(expected_tools)) / len(expected_tools) if expected_tools else 1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}
```

### 2b. 步骤效率（Step Count）

```python
# 统计每次任务的工具调用次数
tool_call_count = sum(
    1 for msg in messages
    if hasattr(msg, "tool_calls") and msg.tool_calls
)
# 目标：同等质量下，步骤越少越好
```

### 2c. LLM-as-Judge 轨迹评分

```python
TRAJECTORY_JUDGE_PROMPT = """
你是一位 AI Agent 评审专家。请根据以下信息对 Agent 的工具调用轨迹打分。

用户指令：{instruction}

Agent 调用轨迹：
{tool_calls}

最终输出：
{output}

请从以下维度各打 1-5 分：
1. 工具选择合理性：是否选择了正确的工具？
2. 调用效率：有没有冗余或重复调用？
3. 输出完整性：最终输出是否完整回答了用户指令？
4. 忠实性：输出内容是否基于真实的邮件内容，没有编造？

请输出 JSON：{{"tool_selection": N, "efficiency": N, "completeness": N, "faithfulness": N, "comment": "..."}}
"""
```

---

## 三、结构化输出层：程序化断言 + LLM-as-Judge

Agent 输出是 `AgentOutput` Pydantic 对象，可做两类检验。

### 3a. 格式/规则断言

```python
def assert_output_format(output: AgentOutput):
    # 优先级格式
    valid_priorities = {"P0", "P1", "P2", "P3"}
    for todo in output.todos:
        assert todo.priority in valid_priorities, f"非法优先级: {todo.priority}"

    # 截止日期格式（如果有）
    from datetime import datetime
    for todo in output.todos:
        if todo.deadline:
            try:
                datetime.fromisoformat(todo.deadline)
            except ValueError:
                raise AssertionError(f"截止日期格式错误: {todo.deadline}")

    # 草稿回复必须有收件人
    for draft in output.draft_replies:
        assert draft.to_uid, "草拟回复缺少目标 UID"

    # 摘要不能为空
    assert output.summary and len(output.summary) > 10, "摘要过短或为空"
```

### 3b. 内容质量（LLM-as-Judge）

```python
OUTPUT_QUALITY_PROMPT = """
以下是原始邮件内容：
{email_content}

以下是 Agent 提取的待办事项：
{todos}

请评估（1-5 分）：
1. 完整性：重要的待办是否都被提取了？
2. 准确性：待办内容是否准确反映了邮件要求？
3. 优先级合理性：P0/P1/P2 的划分是否合理？

输出 JSON：{{"completeness": N, "accuracy": N, "priority": N}}
"""
```

---

## 四、Query Rewrite 层

评估 `rewrite_node` 是否有效提升了多轮对话的准确性。

### 4a. 语义保真度（改写前后意图一致性）

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(text1: str, text2: str) -> float:
    emb1, emb2 = model.encode([text1, text2])
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

# 改写前后的语义相似度应 >= 0.75
similarity = semantic_similarity(original_instruction, rewritten_instruction)
assert similarity >= 0.75, f"改写偏离原意：{similarity:.2f}"
```

### 4b. 检索效果 Delta（Context Recall 差值）

```
ΔRAG = Context_Recall(改写后指令) - Context_Recall(原始指令)
```

- ΔRAG > 0：改写有效提升检索
- ΔRAG ≈ 0：改写无害但无益（首轮透传是正确的）
- ΔRAG < 0：改写偏离了原始意图（需调整 prompt）

### 4c. 端到端 A/B 对比

| 配置 | 多轮对话准确率 | 备注 |
|------|--------------|------|
| 无 rewrite_node（直接透传）| - | baseline |
| 有 rewrite_node | - | 本项目实现 |

测试场景：构造含代词/省略的追问（如"这个作业"、"他说的那个"），比较两种配置下 LLM-as-Judge 的打分。

---

## 五、端到端黄金测试集

整合以上各层，构造完整的回归测试：

```python
GOLDEN_CASES = [
    {
        "id": "tc001",
        "description": "单封 P0 邮件，含明确截止日期",
        "emails": [...],                          # 邮件列表（EmailItem）
        "instruction": "生成我的待办",
        "expected": {
            "has_p0": True,
            "deadline_present": True,
            "tools_called": ["fetch_email_detail"],
        },
    },
    {
        "id": "tc002",
        "description": "多轮对话：追问截止日期",
        "conversation": [
            {"instruction": "根据邮件生成待办"},
            {"instruction": "这个作业什么时候截止？"},  # 含代词，测 rewrite
        ],
        "expected": {
            "rewrite_triggered": True,
            "answer_contains_deadline": True,
        },
    },
    {
        "id": "tc003",
        "description": "RAG 历史检索：跨越当前批次",
        "instruction": "关于项目A上周的进展",
        "expected_tools": ["search_emails_by_topic"],
        "expected_context_recall": 0.8,
    },
]
```

---

## 六、评估优先级建议

| 评估层 | 难度 | 价值 | 建议优先级 |
|--------|------|------|-----------|
| 结构化输出格式断言 | 低 | 高（回归保护）| ⭐⭐⭐ 立即实现 |
| LLM-as-Judge 输出质量 | 低 | 高 | ⭐⭐⭐ 立即实现 |
| Tool F1 工具调用准确率 | 中 | 高 | ⭐⭐ 人工标注后实现 |
| RAGAS 检索指标 | 中 | 中（需 ground truth）| ⭐⭐ 有标注数据后实现 |
| Query Rewrite Delta | 中 | 中 | ⭐⭐ A/B 测试时实现 |
| 端到端黄金测试集 | 高 | 最高（完整覆盖）| ⭐ 迭代后期实现 |


章节	内容
一、RAG 检索层	RAGAS 四指标、最小运行示例、三级检索消融对比表格
二、Agent 轨迹层	Tool F1 计算代码、步骤效率统计、LLM-as-Judge 评分 prompt
三、结构化输出层	Pydantic 格式断言代码、内容质量 LLM-as-Judge prompt
四、Query Rewrite 层	语义保真度（Embedding 余弦）、检索 Delta、A/B 对比表
五、黄金测试集	端到端用例结构模板，含多轮对话场景
六、优先级建议	按难度/价值排序，指导实现顺序

![alt text](image.png)