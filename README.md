# Email Agent

一个基于 **LangGraph ReAct 架构**的智能邮件处理 Agent，通过 IMAP 拉取邮件，使用 Embedding 语义聚类对邮件分组，并以 LLM（DeepSeek）驱动 Agent 图完成待办提取、会议识别、草稿起草、跟进追踪等任务，最终输出结构化 Markdown 报告。

## 功能特性

- 📧 **IMAP 邮件拉取**：支持 IMAP 协议，UID 增量同步，避免重复处理
- 🎯 **两阶段智能聚类**
  - 预聚类：主题指纹（剥离 Re/Fwd/票号）+ 发件人域 + 时间窗
  - 细化合并：`sentence-transformers/all-MiniLM-L6-v2` 本地 Embedding + 余弦相似度（TF-IDF 兜底）
- 🤖 **LangGraph ReAct Agent**：StateGraph 驱动，节点链路 fetch → cluster → **rewrite** → plan → act → summarize → reflect，支持工具调用循环与自反思修正
- 🔍 **RAG 历史邮件检索**：FAISS 向量库 + BM25 稀疏检索 + RRF 倒排融合，通过 Message-ID / References 还原完整对话链
- 🛠️ **Agent 工具集**：`fetch_email_detail` / `search_emails_by_topic` / `check_pending_replies` / `extract_meeting_info` / `draft_reply` / `generate_report` 等
- 📋 **丰富结构化输出**：待办（含优先级/截止日期）、会议日程、跟进事项、草拟回复、VIP 邮件摘要
- 🧠 **多轮对话记忆**：SQLite Checkpointer 持久化 LangGraph 状态，`--session` 参数区分会话
- 💬 **交互式对话模式**：`--chat` 参数开启持续会话，邮件只拉取一次，后续问题基于同一批邮件追问
- ⚡ **预设模式**：`morning-briefing` / `follow-up-check` / `weekly-report` 一键使用

## 技术栈

| 层次 | 技术 |
|------|------|
| Agent 框架 | LangGraph StateGraph + SQLite Checkpointer |
| LLM | DeepSeek（`deepseek-chat`，OpenAI 兼容 API，function_calling 结构化输出）|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`（本地推理，无需云端 API）|
| RAG 检索 | FAISS（密集）+ BM25（稀疏）+ RRF 融合 + rerank（默认关闭） |
| IMAP | imapclient + Python 3.14 兼容适配 |
| 聚类兜底 | scikit-learn TF-IDF + 余弦相似度 + 并查集 |
| 其他 | python-dotenv · tenacity · beautifulsoup4 |

## 系统架构

```
IMAP 邮件
    │
    ▼
fetch_node ──► (RAG 历史上下文注入)──► 写入 FAISS 向量库
    │
    ▼
cluster_node  (Embedding 聚类 → TF-IDF 兜底)
    │
    ▼
rewrite_node  (Query Rewrite：基于对话历史改写模糊指令)
    │         首轮直接透传，后续轮 LLM 补全代词/省略条件
    ▼
plan_node ◄────────────────────────── reflect_node ◄─────┐
    │  (llm_with_tools)                  │ (自评/修正)      │
    │                                     └─────────────────┤
    ├──► act_node (工具调用) ──────────────────────────► plan_node
    │
    └──► summarize_node
              │
              ▼
          AgentOutput (todos / meetings / follow_ups / draft_replies / vip_alerts)
              │
              ▼
         Markdown 报告 / JSON
```

## 安装

### 1. 克隆项目

```bash
git clone <repository-url>
cd emailAgent
```

### 2. 安装依赖（推荐 uv）

```bash
# 安装 uv（Windows PowerShell）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 创建虚拟环境并安装所有依赖
uv sync

# 激活虚拟环境
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
```

**或使用 pip：**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **注意**：首次运行时会自动从 HuggingFace 下载 `all-MiniLM-L6-v2` 模型（约 90MB），之后缓存至本地，无需重复下载。

## 配置

在项目根目录创建 `.env` 文件（文件名为 `.env`，不要加 `.txt` 扩展名）：

```env
# IMAP 邮箱配置（必填）
IMAP_HOST=imap.example.com
IMAP_PORT=993
IMAP_USER=your_email@example.com
IMAP_PASSWORD=your_app_password
MAILBOX=INBOX

# DeepSeek API（必填）
DEEPSEEK_API_KEY=sk-xxx

# DeepSeek 可选配置（有默认值）
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# 聚类参数（可选）
TIME_WINDOW_HOURS=72     # 预聚类时间窗（小时），默认 72
SIM_THRESHOLD=0.55       # 相似度合并阈值（0-1），默认 0.55

# Agent 参数（可选）
AGENT_MAX_ITERATIONS=10  # 最大工具调用轮次，默认 10
EMAIL_STORE_PATH=email_vector_store   # FAISS 向量库存储目录
CHECKPOINT_DB_PATH=agent_memory.db   # SQLite 对话记忆数据库

# 其他（可选）
STATE_PATH=imap_state.json  # 增量同步状态文件
REQUEST_TIMEOUT=60
```

### 常见邮箱 IMAP 配置

| 邮箱服务 | IMAP_HOST | 备注 |
|---------|-----------|------|
| Gmail | `imap.gmail.com` | 需开启两步验证，使用[应用专用密码](https://support.google.com/accounts/answer/185833) |
| Outlook/Office 365 | `outlook.office365.com` | 使用账户密码或应用密码 |
| QQ 邮箱 | `imap.qq.com` | 在设置中开启 IMAP，使用[授权码](https://service.mail.qq.com/cgi-bin/help?subtype=1&&id=28&&no=1001256) |
| 163 邮箱 | `imap.163.com` | 开启 IMAP 并使用授权码 |
| iCloud | `imap.mail.me.com` | 使用[应用专用密码](https://support.apple.com/zh-cn/102654) |

## 使用方法

### 基本用法

```bash
python -m src.cli --since 7d
```

### 完整参数示例

```bash
python -m src.cli \
  --since 7d \
  --mailbox INBOX \
  --instruction "根据邮件生成我的待办并按优先级排序" \
  --session my-work \
  --output out/report.md
```

### 命令行参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--since` | 抓取起始时间（`7d` / `48h` / `2025-01-01`）| `7d` |
| `--mailbox` | IMAP 邮箱文件夹 | 配置中的 `MAILBOX` |
| `--instruction` | 给 Agent 的自然语言指令 | `"根据邮件生成我的待办并按优先级排序"` |
| `--mode` | 预设模式（见下方） | 无 |
| `--days` | `follow-up-check` 模式下超过几天未回复视为待跟进 | `3` |
| `--session` | 对话 Session ID，同 ID 下保留多轮上下文 | `default` |
| `--output` | 输出 Markdown 文件路径，不填仅打印控制台 | 无 |
| `--json` | 输出结构化 JSON（AgentOutput 原始格式）| 关闭 |
| `--rescan` | 忽略增量状态，重新抓取所有符合条件邮件 | 关闭 |
| `--include-seen` | 包含已读邮件 | 关闭 |
| `--chat` | 交互式多轮对话模式，程序持续运行等待追问（输入 `exit` 退出）| 关闭 |

### 交互式多轮对话（`--chat`）

```bash
# 启动交互式会话，邮件只拉取一次，可持续追问
python -m src.cli --since 7d --include-seen --chat --session my-session
```

```
[Chat] 交互式对话模式启动（session=my-session）
[Chat] 邮件将在第一次指令时拉取，后续对话复用同一批邮件
[Chat] 输入 exit / quit / q 退出
>>> 根据邮件生成我的待办
...
>>> 把 P0 的任务详细说说
...（Agent 自动将 "P0 的任务" 改写为完整指令，基于对话历史语义理解）
>>> exit
```

### 预设模式（`--mode`）

```bash
# 每日早间速览：按紧急度排序 + 提取会议 + 标注 VIP 邮件
python -m src.cli --mode morning-briefing --since 1d

# 跟进检查：找出超过 N 天未回复的邮件
python -m src.cli --mode follow-up-check --days 3 --since 14d

# 周报生成：主要决策 + 待跟进 + 下周关注
python -m src.cli --mode weekly-report --since 7d
```

### 自定义指令示例

```bash
# 提取会议议题
python -m src.cli --instruction "提取需要会议讨论的议题，给出参会人建议与会前准备清单"

# 识别风险与阻塞
python -m src.cli --instruction "先列出阻塞他人的事项，其次列潜在风险与依赖，再给一般待办"

# 按主题分组
python -m src.cli --instruction "按主题指纹或参与者分组，每组先给 1 句组摘要，再列待办"

# 草拟回复
python -m src.cli --instruction "为每个关键线程生成中文回复草稿（称呼+要点+行动+礼貌结尾）"
```

### 多轮对话（跨次运行保留上下文）

```bash
# 第一次：生成待办
python -m src.cli --session work-q1 --since 7d --instruction "生成待办"

# 第二次（同 session）：追问，Agent 可访问上一轮状态
python -m src.cli --session work-q1 --instruction "把上面 P1 的待办提取出来发给我看"
```

## 输出格式

### Markdown 报告（默认）

```markdown
# 邮件助理报告

_生成时间：2026-03-03 01:18_

## 总结
...

## ⭐ VIP 邮件
- ...

## 待办事项
- **[P1]** 评估 Notion 企业搜索（截止 2026-03-09）→ 详细阅读功能介绍

## 会议日程
- **项目评审** | 时间：2026-03-05 14:00 | 地点：腾讯会议

## 跟进事项
- [我方发起] 发给张三的方案确认（2026-03-06）

## 草拟回复（待确认）
### 回复 UID=123（语气：formal）
...
```

### 结构化 JSON（`--json`）

```bash
python -m src.cli --json --since 7d
```

输出 `AgentOutput` Pydantic 模型的完整 JSON：

```json
{
  "todos": [
    {"title": "...", "priority": "P1", "deadline": "2026-03-09", "suggested_action": "..."}
  ],
  "meetings": [],
  "follow_ups": [],
  "draft_replies": [],
  "vip_alerts": [],
  "summary": "...",
  "unresolved": []
}
```

## 工作原理

### 1. 邮件拉取（`fetch_node`）

- IMAP `UNSEEN + SINCE` 搜索，UID 增量过滤
- RAG 历史上下文注入：通过 Message-ID / References 链路找到父邮件，补充 `historical_context` 字段
- 新邮件异步写入 FAISS 向量库（供后续 RAG 检索）

### 2. 智能聚类（`cluster_node`）

**预聚类（粗分组）**

- 主题指纹：去除 `Re:` / `Fwd:` 前缀和 `[TICKET-123]` 票号，转小写
- 发件人域：提取邮箱域名
- 时间窗：默认 72 小时内同主题默认归组

**细化合并**

- 优先使用 `all-MiniLM-L6-v2` 计算 Embedding 余弦相似度
- 失败时降级 TF-IDF 余弦相似度
- 并查集（Union-Find）高效合并，阈值默认 0.55

### 3. ReAct Agent（`plan_node` + `act_node`）

- LangGraph StateGraph，plan 节点决定调用工具还是直接总结
- 可用工具：

| 工具 | 类型 | 说明 |
|------|------|------|
| `fetch_email_detail` | 只读 | 获取指定 UID 完整正文 |
| `search_emails_by_topic` | 只读 | RAG 语义搜索历史邮件 |
| `check_pending_replies` | 只读 | 找出超 N 天未回复邮件 |
| `extract_meeting_info` | 只读 | 提取会议时间/地点/参与者 |
| `draft_reply` | 生成类 | 起草回复草稿（不自动发送）|
| `draft_new_email` | 生成类 | 起草新邮件（不自动发送）|
| `adjust_tone` | 生成类 | 调整草稿语气 |
| `generate_report` | 生成类 | 基于 RAG 历史邮件生成报告 |
| `send_email` | 写操作 | 发送邮件（需人工确认）|
| `flag_email_as_important` | 写操作 | 标记重要邮件 |

### 4. 自反思（`reflect_node`）

- summarize 完成后自动进入 reflect 节点
- LLM 评审输出质量，若不满意则将 critique 注入下一轮 plan 节点重新生成
- 配合 `reflection_count` 防止无限循环

### 5. 多轮记忆（SQLite Checkpointer）

- 基于 LangGraph SQLite Checkpointer 持久化所有节点状态
- 通过 `--session` 区分不同工作上下文
- 删除 `agent_memory.db` 可清空全部会话历史

## 项目结构

```
emailAgent/
├── src/
│   ├── agent/
│   │   ├── graph.py         # LangGraph StateGraph 主图定义
│   │   ├── nodes.py         # 所有图节点（fetch/cluster/rewrite/plan/act/summarize/reflect）
│   │   ├── state.py         # AgentState TypedDict 定义（含 rewritten_instruction 字段）
│   │   ├── structured.py    # AgentOutput Pydantic 模型（结构化输出）
│   │   └── tools.py         # Agent 工具集定义与实现
│   ├── cli.py               # 命令行入口
│   ├── config.py            # 配置加载（.env + 环境变量）
│   ├── imap_client.py       # IMAP 邮件拉取（含 Python 3.14 兼容补丁）
│   ├── clusterer.py         # 两阶段聚类（Embedding 优先 + TF-IDF 兜底）
│   ├── email_store.py       # RAG 向量库（FAISS + BM25 + RRF 混合检索）
│   ├── embeddings.py        # 本地 HuggingFace Embedding 封装
│   ├── llm_summarizer.py    # 结构化摘要（function_calling 方式）
│   ├── models.py            # EmailItem / EmailThread 数据模型
│   ├── renderer.py          # Markdown 渲染输出
│   ├── state_store.py       # UID 增量同步状态管理
│   └── text_utils.py        # 文本预处理工具
├── .env                     # 环境变量配置（需自行创建，勿提交 git）
├── imap_state.json          # 增量同步状态（自动生成）
├── agent_memory.db          # SQLite 对话记忆（自动生成）
├── email_vector_store/      # FAISS 向量库（自动生成）
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 常见问题

### Q: 提示 "Missing environment variable: IMAP_HOST"

`.env` 文件需位于项目根目录（与 `src/` 同级），文件名为 `.env`（不是 `.env.txt`），格式为 `KEY=VALUE`。

### Q: 首次运行很慢

首次运行会下载 `sentence-transformers/all-MiniLM-L6-v2` 模型（约 90MB）到本地缓存，之后启动正常。

### Q: 没有显示任何邮件

1. 尝试 `--since 30d` 拉取更长时间段
2. 加 `--include-seen` 包含已读邮件
3. 加 `--rescan` 忽略增量状态重新抓取
4. 删除 `imap_state.json` 清空增量记录

### Q: 如何完全重置（清空所有历史状态）

```powershell
# Windows PowerShell
Remove-Item -ErrorAction SilentlyContinue .\imap_state.json
Remove-Item -ErrorAction SilentlyContinue .\agent_memory.db
Remove-Item -ErrorAction SilentlyContinue -Recurse .\email_vector_store
```

### Q: IMAP 登录失败

1. 确认已在邮箱设置中开启 IMAP 服务
2. Gmail / QQ / 163 需使用**应用专用密码/授权码**，而非普通登录密码
3. 确认 `IMAP_HOST` 和 `IMAP_PORT`（通常为 993）正确

### Q: 相似度阈值如何调整？

在 `.env` 中设置 `SIM_THRESHOLD`：
- 偏高（如 `0.7`）：更严格，减少误合并
- 偏低（如 `0.4`）：更宽松，可能合并不相关邮件
- 默认 `0.55` 是平衡值

### Q: 支持其他 LLM 服务吗？

支持所有 OpenAI 兼容 API，修改 `.env` 中的 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL` 即可。**注意**：LLM 必须支持 Function Calling（工具调用），否则结构化输出会失败。
