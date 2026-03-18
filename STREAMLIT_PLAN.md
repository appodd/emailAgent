# Streamlit UI 实现方案

> 目标：为 emailAgent 提供可视化交互界面，核心体验是多轮对话 + AgentOutput 结构化展示。

---

## 一、技术架构

```
app.py                          ← Streamlit 入口
    │
    ├── st.session_state        ← 跨组件共享状态
    │       ├── graph           ← LangGraph CompiledGraph（懒初始化，单例）
    │       ├── config          ← Config 实例
    │       ├── thread_id       ← LangGraph checkpoint session ID
    │       ├── email_items     ← 已抓取邮件列表（拉取一次后复用）
    │       ├── threads         ← cluster_node 输出
    │       ├── chat_history    ← [(role, content, meta)] 展示用历史
    │       └── last_output     ← 最近一次 AgentOutput
    │
    ├── pages/
    │       ├── 1_💬_对话.py     ← 主页：多轮交互式对话
    │       ├── 2_🗂️_聚类.py     ← 邮件聚类可视化
    │       ├── 3_🔍_搜索.py     ← RAG 历史邮件搜索
    │       └── 4_⚙️_设置.py     ← 参数配置面板
    │
    └── ui/
            ├── components.py   ← 可复用 UI 组件（AgentOutput 渲染等）
            └── agent_runner.py ← 封装 graph.stream()，处理状态更新
```

---

## 二、页面详细设计

### 主页 `app.py` / `pages/1_💬_对话.py` — 交互式对话（P0）

**布局**：左侧边栏（配置快捷项） + 主区域（对话界面）

**核心交互流程**：
```
用户输入指令
    │
    ▼
agent_runner.run(instruction, session_state)
    │   内部：graph.stream() 逐节点推进
    │   实时更新进度提示：
    │     "📥 抓取邮件..." → "🗂️ 聚类中..." →
    │     "✏️ Query Rewrite..." → "🔍 检索中..." → "🤖 规划中..."
    ▼
展示结果：
  ├─ Query Rewrite 折叠块：原始指令 / 改写结果 / sub_queries
  ├─ RAG 召回折叠块：retrieved_context 中的邮件片段
  └─ AgentOutput 渲染（见下）
```

**对话气泡样式**：
- Human：右对齐，蓝色背景
- AI：左对齐，灰色背景，含 AgentOutput 结构化块

**初始化时机**：首次发送时 lazy init graph，之后复用（不重复拉邮件）。

---

### `pages/2_🗂️_聚类.py` — 聚类可视化（P1）

展示 `cluster_node` 对当次抓取邮件的分组结果：

- 顶部统计：邮件总数 / 线程数 / 聚类算法（Embedding / TF-IDF 兜底）
- 每个 EmailThread 折叠卡片：
  ```
  ▼ 【项目进展讨论】  3 封邮件  参与者: alice@x.com, bob@y.com
      ├─ 2026-03-10  From: alice@x.com  "Hi, 关于..."
      ├─ 2026-03-11  From: bob@y.com    "收到，我们..."
      └─ 2026-03-12  From: alice@x.com  "好的，那就..."
  ```
- 支持按"主题"/"参与者"/"时间"排序

---

### `pages/3_🔍_搜索.py` — RAG 历史邮件搜索（P1）

直接调用 `build_hybrid_retriever` 做语义搜索，无需走完整 Agent 图：

- 搜索框 + 搜索按钮
- 显示 sub_queries（自动拆分多实体查询）
- 结果列表：
  ```
  得分: 0.82 | 2026-01-15 | From: alice@x.com | Subject: Q4 财务汇报
  "请查收 Q4 各部门财务数据汇总，附件见..."
  [展开全文]
  ```

---

### `pages/4_⚙️_设置.py` — 配置面板（P2）

运行时修改参数，写入 st.session_state.config：

| 分组 | 参数 |
|------|------|
| IMAP 连接 | HOST / PORT / USER / MAILBOX |
| LLM | DeepSeek API Key / Model / Base URL |
| 检索 | RAG BM25 Weight / Rerank 开关 / Rerank Top-N |
| Agent | Reflection 开关 / 抓取天数 / Session ID |

---

## 三、`ui/components.py` — AgentOutput 渲染

```python
render_todos(todos)         # 优先级标签（红/橙/绿） + 表格
render_meetings(meetings)   # 日历卡片：时间/地点/参与者
render_follow_ups(items)    # sent/received 分组列表
render_draft_replies(drafts)# 代码块展示草稿 + "确认发送"按钮（disabled）
render_vip_alerts(alerts)   # 橙色 warning 框
render_summary(summary)     # 顶部 info 框
```

---

## 四、`ui/agent_runner.py` — Graph 执行封装

```python
def run_agent(instruction, session_state, is_first_turn):
    """
    封装 graph.stream()，实时更新 st.session_state：
      - 每个节点完成时 yield 节点名（供 UI 显示进度）
      - 结束后从 graph.get_state() 读取最终 output
    """
```

核心要点：
- `graph.stream()` 用 `stream_mode="values"` 逐节点推进
- `st.status()` 展示节点进度（LangGraph 节点名 → 中文描述）
- 首轮传完整 initial_state，后续轮只传 `instruction` 等必要字段

---

## 五、文件变更清单

```
新增：
  app.py                       Streamlit 入口（侧边栏导航 + session 初始化）
  pages/
    1_💬_对话.py
    2_🗂️_聚类.py
    3_🔍_搜索.py
    4_⚙️_设置.py
  ui/
    __init__.py
    components.py
    agent_runner.py

不修改：
  src/ 下所有现有文件（Agent 逻辑保持不变）
```

---

## 六、实施优先级

| 步骤 | 内容 | 优先级 |
|------|------|--------|
| 1 | `ui/agent_runner.py`：封装 graph.stream() | P0，其他页面依赖 |
| 2 | `ui/components.py`：AgentOutput 渲染组件 | P0，对话页依赖 |
| 3 | `app.py` + `pages/1_💬_对话.py`：对话主页 | P0，核心体验 |
| 4 | `pages/2_🗂️_聚类.py`：聚类可视化 | P1 |
| 5 | `pages/3_🔍_搜索.py`：RAG 搜索 | P1 |
| 6 | `pages/4_⚙️_设置.py`：配置面板 | P2 |

---

## 七、依赖

```
streamlit>=1.32.0    # st.status() 需要 1.28+
```

在 `requirements.txt` 追加即可，无其他新依赖（LangGraph / FAISS 已有）。
