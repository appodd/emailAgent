# Email Agent

一个智能邮件处理工具，通过 IMAP 拉取未读邮件，使用智能聚类算法将相关邮件分组，并利用 LLM（DeepSeek）生成优先级待办清单或邮件摘要。

## 功能特性

- 📧 **IMAP 邮件拉取**：支持通过 IMAP 协议拉取未读邮件
- 🔄 **增量同步**：基于 UID 的增量处理，避免重复处理已读邮件
- 🎯 **智能聚类**：两阶段聚类算法
  - **预聚类**：基于主题指纹（去除 Re/Fwd/票号）+ 发件人域 + 时间窗
  - **细化合并**：使用 TF-IDF + 余弦相似度进一步合并相关邮件
- 🤖 **LLM 总结**：使用 DeepSeek（OpenAI 兼容 API）生成待办清单或摘要
- 📝 **多格式输出**：支持控制台输出和 Markdown 文件导出
- ⚙️ **灵活配置**：支持环境变量和 `.env` 文件配置

## 安装

### 1. 克隆或下载项目

```bash
git clone <repository-url>
cd emailAgent
```

### 2. 安装 uv（推荐）

`uv` 是一个极速的 Python 包管理器和项目管理工具，推荐使用它来管理项目环境。

**安装 uv：**

**Windows (PowerShell)：**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS：**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**或使用 pip：**
```bash
pip install uv
```

### 3. 使用 uv 创建虚拟环境并安装依赖（推荐）

使用 `uv` 可以一步完成虚拟环境创建和依赖安装：

```bash
# 创建虚拟环境并安装所有依赖
# 进入到项目目录中
uv sync

# 激活虚拟环境（uv 会自动管理）
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows CMD
.\.venv\Scripts\activate.bat
# Linux/macOS
source .venv/bin/activate
```

**或者使用 uv 直接运行（无需手动激活虚拟环境）：**
```bash
# 使用 uv run 直接运行，uv 会自动管理虚拟环境
uv run python -m src.cli --since 7d
```

### 4. 传统方式（备选）

**使用 venv：**
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows CMD
.\.venv\Scripts\activate.bat
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

**使用 conda：**
```bash
conda create -n emailAgent python=3.10 -y
conda activate emailAgent
pip install -r requirements.txt
```

## 配置

### 必需配置

创建项目根目录下的 `.env` 文件（**注意：文件名就是 `.env`，不要加 `.txt` 扩展名**）：

```env
IMAP_HOST=imap.example.com
IMAP_PORT=993
IMAP_USER=your_email@example.com
IMAP_PASSWORD=your_app_password
MAILBOX=INBOX
DEEPSEEK_API_KEY=sk-xxx
```

### 常见邮箱 IMAP 配置

| 邮箱服务 | IMAP_HOST | 说明 |
|---------|-----------|------|
| Gmail | `imap.gmail.com` | 需开启两步验证并使用[应用专用密码](https://support.google.com/accounts/answer/185833) |
| Outlook/Office 365 | `outlook.office365.com` | 使用账户密码或应用密码 |
| QQ 邮箱 | `imap.qq.com` | 需在设置中开启 IMAP，使用[授权码](https://service.mail.qq.com/cgi-bin/help?subtype=1&&id=28&&no=1001256) |
| 163 邮箱 | `imap.163.com` | 需开启 IMAP 并使用授权码 |
| iCloud | `imap.mail.me.com` | 需使用[应用专用密码](https://support.apple.com/zh-cn/102654) |

### 可选配置

```env
# DeepSeek API 配置
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# 聚类参数
TIME_WINDOW_HOURS=72          # 时间窗（小时），默认 72
SIM_THRESHOLD=0.55             # 相似度阈值（0-1），默认 0.55，值越高越严格

# 其他配置
STATE_PATH=imap_state.json     # 状态文件路径，默认 imap_state.json
REQUEST_TIMEOUT=60             # 请求超时（秒），默认 60
```

### 环境变量方式（临时配置）

**Windows PowerShell：**
```powershell
$env:IMAP_HOST="imap.gmail.com"
$env:IMAP_PORT="993"
$env:IMAP_USER="your_email@gmail.com"
$env:IMAP_PASSWORD="your_app_password"
$env:MAILBOX="INBOX"
$env:DEEPSEEK_API_KEY="sk-xxx"
```

**Linux/macOS：**
```bash
export IMAP_HOST="imap.gmail.com"
export IMAP_PORT="993"
export IMAP_USER="your_email@gmail.com"
export IMAP_PASSWORD="your_app_password"
export MAILBOX="INBOX"
export DEEPSEEK_API_KEY="sk-xxx"
```

## 使用方法

### 基本用法

```bash
python -m src.cli --since 7d --mailbox INBOX --output out/todo.md --instruction "根据邮件生成我的待办并按优先级排序"
```

### 命令行参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--since` | 抓取起始时间 | `7d` | `7d`（7天）、`48h`（48小时）、`2025-01-01`（绝对日期） |
| `--mailbox` | IMAP 邮箱文件夹 | 使用配置中的 `MAILBOX` | `INBOX`、`[Gmail]/All Mail` |
| `--output` | 输出 Markdown 文件路径 | 无（仅控制台输出） | `out/todo.md` |
| `--instruction` | 给 LLM 的自然语言指令 | `"根据邮件生成我的待办并按优先级排序"` | 见下方示例 |

### 指令示例

不同的 `--instruction` 可以生成不同格式的输出：

```bash
# 生成待办清单（默认）
python -m src.cli --instruction "根据邮件生成我的待办并按优先级排序"

# 生成回复草稿
python -m src.cli --instruction "为每个关键线程生成一段中文回复草稿（称呼+要点+行动+礼貌结尾），并给 3 条要点式待办"

# 提取会议议题
python -m src.cli --instruction "提取需要会议讨论的议题，给出参会人建议与会前准备清单"

# 识别风险与阻塞
python -m src.cli --instruction "先列出阻塞他人的事项，其次列潜在风险与依赖，再给一般待办"

# 按主题分组输出
python -m src.cli --instruction "按主题指纹或参与者分组，每组先给 1 句组摘要，再列待办"
```

## 工作原理

### 1. 邮件拉取

- 通过 IMAP 协议连接邮箱服务器
- 搜索指定时间范围内的未读邮件（`UNSEEN` + `SINCE`）
- 基于 `last_seen_uid` 进行增量过滤，只处理新邮件
- 提取邮件主题、发件人、正文等关键信息

### 2. 智能聚类

#### 预聚类（粗分组）
- **主题指纹**：去除 `Re:`、`Fwd:` 等前缀和票号（如 `[TICKET-123]`），统一转小写
- **发件人域**：提取发件人邮箱域名
- **时间窗**：将邮件按时间窗口（默认 72 小时）分组
- 相同“主题指纹 + 发件人域 + 时间窗”的邮件进入同一初始组

#### 细化合并（TF-IDF + 余弦相似度）
- 对预聚类结果中的邮件计算 TF-IDF 向量
- 使用余弦相似度计算邮件之间的相似度
- 相似度 ≥ 阈值（默认 0.55）的邮件合并为同一线程
- 使用并查集（Union-Find）算法高效合并

### 3. LLM 总结

- 将聚类后的邮件线程格式化
- 调用 DeepSeek API 生成待办清单或摘要
- 支持自定义指令，灵活控制输出格式

### 4. 状态管理

程序会在项目根目录生成 `imap_state.json` 文件，记录：
- `last_seen_uid`：每个邮箱文件夹已处理到的最大 UID
- `processed_uids`：近期已处理过的 UID 列表（用于去重）

**注意**：删除 `imap_state.json` 可以重新处理所有邮件。

## 项目结构

```
emailAgent/
├── src/
│   ├── __init__.py
│   ├── cli.py              # 命令行入口
│   ├── config.py            # 配置加载
│   ├── imap_client.py       # IMAP 邮件拉取
│   ├── clusterer.py          # 聚类算法
│   ├── llm_summarizer.py    # LLM 总结
│   ├── models.py            # 数据模型
│   ├── renderer.py          # 输出渲染
│   ├── state_store.py       # 状态管理
│   └── text_utils.py        # 文本处理工具
├── .env                     # 环境变量配置（需自行创建）
├── imap_state.json          # 增量同步状态（自动生成）
├── requirements.txt         # Python 依赖
└── README.md               # 本文件
```

## 常见问题

### Q: 提示 "Missing environment variable: IMAP_HOST"

**A:** 确保 `.env` 文件：
- 位于项目根目录（与 `src` 文件夹同级）
- 文件名是 `.env`（不是 `.env.txt`）
- 格式正确：`KEY=VALUE`（不要有 `$env:` 前缀）
- 每行一个配置，不要在同一行写注释

### Q: 显示 "没有新的未读邮件"

**A:** 可能原因：
1. 确实没有未读邮件：尝试放宽时间范围 `--since 30d`
2. 邮件已被处理：删除 `imap_state.json` 重新处理
3. 邮件是已读状态：手动将邮件标记为未读
4. 邮箱文件夹名称错误：确认 `--mailbox` 参数正确

### Q: IMAP 登录失败

**A:** 检查：
1. 是否开启了 IMAP 服务（邮箱设置中）
2. 是否使用了正确的应用专用密码/授权码（不是普通登录密码）
3. IMAP 服务器地址和端口是否正确
4. 是否触发了邮箱安全限制（可能需要允许"不够安全的应用"）

### Q: 如何重新处理所有邮件？

**A:** 删除或重命名 `imap_state.json` 文件：
```bash
# Windows PowerShell
Remove-Item .\imap_state.json

# Linux/macOS
rm imap_state.json
```

### Q: 相似度阈值如何调整？

**A:** 在 `.env` 文件中设置 `SIM_THRESHOLD`：
- 值越高（如 0.7）：更严格，减少误合并
- 值越低（如 0.4）：更宽松，可能合并不相关邮件
- 默认 0.55 是平衡值

### Q: 支持哪些 LLM 服务？

**A:** 项目使用 OpenAI 兼容的 API，理论上支持：
- DeepSeek（默认）
- OpenAI
- 其他兼容 OpenAI API 格式的服务

只需修改 `.env` 中的 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。

## 技术栈

- **Python 3.10+**
- **imapclient**：IMAP 协议客户端
- **scikit-learn**：TF-IDF 向量化和余弦相似度计算
- **beautifulsoup4**：HTML 邮件解析
- **openai**：OpenAI 兼容 API 客户端
- **python-dotenv**：环境变量管理
- **tenacity**：API 调用重试机制

## 许可证

[根据项目实际情况填写]

## 贡献

欢迎提交 Issue 和 Pull Request！
