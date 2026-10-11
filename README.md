# 🧭 Durable Research Agent Harness

<p align="center">
  <img src="docs/images/durable-agent-hero.png" alt="一位研究者与研究 Agent 协作；资料、工具、检查点、审批和报告环绕在旁" width="100%" />
</p>

<p align="center">
  <strong>一个从零实现、可恢复、可审计的研究型 Agent Harness。</strong><br />
  <sub>让模型负责动脑，让 Harness 负责掌舵；聪明也要有刹车。</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/tests-44%20passed-22C55E" alt="44 tests passed" />
  <img src="https://img.shields.io/badge/control%20loop-project--owned-8B5CF6" alt="Project-owned control loop" />
  <img src="https://img.shields.io/badge/agent%20framework-none-F97316" alt="No agent framework" />
</p>

> **一句话概括：** 这是一个会动脑、会等人点头、断电后还能找回进度条的小型研究 Agent。它能异步调用工具、在副作用前等待人工审批、把状态原子落盘，并在新进程中安全恢复。

---

## 🏅 这份作品想证明什么？

我没有把重点放在“再接几个工具”，而是把它当作一次小型 Agent 基础设施设计：**让不完全可靠的模型，运行在边界清晰、可恢复、可复核的系统里。**

| 我的工程亮点 | 在仓库里的落点 | 为什么这很重要 |
|---|---|---|
| 🎮 **控制权留在自己手里** | `AgentLoop` 是项目自写的多轮控制平面；没有让 Agent 框架接管执行语义。 | 模型只能提议，不能自行获得权限或偷偷改流程。 |
| 🧯 **把最难的崩溃窗口摊在桌面上** | `ApprovalService` 在写入前持久化执行栅栏；故障注入测试验证恢复时拒绝盲重试。 | 不把“exactly-once”当口号，而是诚实处理副作用已经发生但结果尚未落盘的真实风险。 |
| ⚡ **异步不是装饰品** | `AsyncToolScheduler` 对独立只读调用并发执行，并保存耗时与失败元数据。 | 能用测试证明并发，也能保证一个网页失败不会带崩同批工具。 |
| 🧱 **把安全写成静态能力，而非提示词愿望** | `ToolSpec`、路径边界、SSRF 防护、审批策略和 schema 校验都在 Harness 侧。 | 无论模型怎么“发挥”，它都不能越过文件、网络和写入权限的护栏。 |
| 🔍 **每个主张都有小票** | `MockLLM` 离线测试、真实 API 冒烟验证、JSONL 轨迹与 `summary.json`。 | 44 项确定性测试把关键行为变成可复验事实，而不是 demo 的一次偶然成功。 |

> **我希望审阅者看到的不是“一个会聊天的脚本”，而是一位知道 Agent 在什么时候最容易失控、并愿意把安全带提前系好的人。**

---

## ✨ 这个小家伙厉害在哪？

很多 Agent 原型的剧情是“模型说一句，工具冲出去”。本项目的主角却是 **Harness**：即使模型、网络或进程偶尔耍小脾气，系统仍应回答三个很现实的问题——现在做到哪一步？刚刚有没有真的写文件？下一步还安全吗？

| 🌈 招式 | 幕后机关 | 所以呢？ |
|---|---|---|
| 🧠 自主多轮 Loop | `AgentLoop` 接收模型的工具提议、执行/回填结果，再决定继续、暂停或完成。 | 不是预先硬编码的静态工作流。 |
| ⚡ 真正异步 | 独立只读工具通过 `AsyncToolScheduler` 并发；单个失败会转为结构化结果。 | 并发、超时、重试与失败隔离都有可测证据。 |
| ✋ Human-in-the-loop | `write_report` 属于写入能力，普通调度器永远不会直接执行。 | 人工明确批准或拒绝副作用。 |
| 💾 可恢复执行 | 待批准调用先写入原子 checkpoint；新的服务实例可只凭 `run_id` 恢复。 | 进程退出不等于任务状态丢失。 |
| 🛡️ 安全边界 | 工具 schema、路径范围、SSRF 防护和审批策略全部由 Harness 固定。 | 模型文本无法扩权。 |
| 🔎 可观测性 | 脱敏 JSONL 轨迹与摘要记录状态、批次、审批和工具元数据。 | 可以复核执行过程，而非相信口头描述。 |

### 我们不卖“exactly-once 魔法”

在“文件已经写入，但写入结果的 checkpoint 尚未落盘”这个无法完全消除的崩溃窗口中，系统会留下持久化执行栅栏，并在恢复时 **拒绝盲目重试**。宁可请人类先看一眼目标文件，也不为了一个听起来很酷的词而偷偷写第二遍。

---

## 🎨 一张图看完小剧场

<p align="center">
  <img src="docs/images/durable-agent-workflow.png" alt="从任务、并发工具、检查点、人工审批到报告和轨迹的可恢复 Agent 工作流" width="100%" />
</p>

1. **模型先举手**：它可以请求读取本地资料或抓取公开网页。
2. **Harness 先验票**：参数不合法会变成模型可见的 `ToolResult`；独立只读工具可以并发。
3. **要写东西？先存档**：一旦出现 `write_report`，系统保存准确调用并转为 `waiting_approval`。
4. **人类最后拍板**：批准时只执行保存的调用；拒绝时把原因回填给模型。
5. **留好小票再收工**：新进程可以加载 checkpoint，`events.jsonl` 与 `summary.json` 留下脱敏证据。

---

## 🗺️ 技术路线图：谁来动脑，谁来踩刹车？

```text
                         ┌────────────────────────────┐
                         │     User / CLI command     │
                         │    run · resume · inspect  │
                         └─────────────┬──────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Project-owned Harness                        │
│                                                                      │
│  AgentLoop ◄──────► LLM adapter (OpenAI-compatible / MockLLM)       │
│      │                                                               │
│      ├── ToolRegistry ──► schema validation + static policy          │
│      ├── AsyncToolScheduler ──► concurrency / timeout / retry        │
│      ├── ApprovalService ──► human decision + retry fence            │
│      ├── StateStore ──► atomic checkpoint.json                       │
│      └── TraceRecorder ──► redacted events.jsonl + summary.json      │
└─────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                  read_file · fetch_url · write_report
```

<p align="center">
  <img src="docs/images/durable-agent-technical-roadmap.png" alt="从命令行、模型提议、控制循环、工具校验和并发调度，到审批、持久化、恢复、报告和轨迹的技术路线图" width="100%" />
</p>

图里没有塞进难读的小字，源码路径在这里乖乖对号入座：终端是 CLI，云朵是 LLM adapter，绿色环形机器人是 `AgentLoop`；下方收纳柜是 `ToolRegistry`；文件夹和地球是 `AsyncToolScheduler` 的并发工具；印章是 `ApprovalService`；蓝色保险箱和回转箭头是 `StateStore`；紫色长卷则是 `TraceRecorder` 的审计轨迹。

| 组件 | 职责 | 模型能否绕过？ |
|---|---|---:|
| `AgentLoop` | 驱动多轮提议、工具结果回填、终止与暂停状态。 | 否 |
| `ToolRegistry` | 发布工具 schema，并校验每一次模型来源参数。 | 否 |
| `AsyncToolScheduler` | 让安全的只读调用并发，隔离超时与失败。 | 否 |
| `ApprovalService` | 对写入调用持久化、审批、恢复及防重复执行。 | 否 |
| `StateStore` / `TraceRecorder` | 形成原子 checkpoint 与脱敏审计证据。 | 否 |

---

## 🚀 五分钟开箱小实验

### 1) 给它装上小背包

要求：**Python 3.11+**。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2) 告诉它去哪里找模型

复制模板后填入你的 OpenAI-compatible 凭据：

```bash
cp .env.example .env
# 编辑 .env：OPENAI_API_KEY、OPENAI_MODEL；可选 OPENAI_BASE_URL
set -a; source .env; set +a
```

`.env` 已被 Git 忽略。不要提交密钥、私密任务正文、未审查网页资料或真实运行 checkpoint。

### 3) 画出它的安全活动区

```bash
mkdir -p demo-workspace/data demo-workspace/reports demo-runs
printf 'Durable execution makes research workflows inspectable.\n' \
  > demo-workspace/data/source.txt
```

工具边界是固定的：`read_file` 只能读 `data/`，`write_report` 只能写 `reports/`。

### 4) 发令，开跑！

```bash
research-agent --runs-dir demo-runs run \
  "Read source.txt, check one public source, then write a concise Markdown brief." \
  --workspace demo-workspace
```

当 CLI 返回以下状态时，说明写入提议已经安全保存、等待人类决定：

```text
waiting_approval: <run_id>
```

### 5) 人类最终拍板

批准已保存的写入：

```bash
research-agent --runs-dir demo-runs resume <run_id> --approve \
  --workspace demo-workspace
```

或者拒绝，并将原因作为结构化结果回传给模型：

```bash
research-agent --runs-dir demo-runs resume <run_id> \
  --reject "请先缩短为三段" --workspace demo-workspace
```

任何时候都可以在 **没有 API Key** 的环境检查本地状态：

```bash
research-agent --runs-dir demo-runs inspect <run_id>
```

---

## 🧰 三件工具，三道护栏

| 工具 | 能力范围 | 关键防护 |
|---|---|---|
| `read_file(path)` | 读取工作区 `data/` 下的 UTF-8 文本。 | 路径边界校验，拒绝借由软链接逃逸。 |
| `fetch_url(url)` | 异步读取受限大小的公开 HTTP(S) 文本。 | 拒绝本地、私网、metadata 地址与不安全重定向。 |
| `write_report(path, content)` | 写入工作区 `reports/` 下已存在的目录。 | 需要审批；普通 scheduler 不能调用；恢复时使用已持久化参数。 |

`ToolSpec` 是策略信任边界：模型的自然语言、工具参数和提示词都不能改变写入审批、并发策略、重试次数或文件系统访问范围。

---

## 🧪 不靠口号，靠测试小票

```bash
PYTHONPATH=src python3.11 -m pytest -q
python3.11 -m ruff check src tests
python3.11 -m compileall -q src tests
```

当前测试集为 **44 项**，全程离线、确定性执行，不需要 API Key。

| 验证场景 | 关键断言 |
|---|---|
| 多轮 Agent Loop | 模型提议、工具结果回填、最终回答形成完整闭环。 |
| 并发与失败隔离 | 本地读取与网页调用真实重叠；网页失败不会取消健康调用。 |
| 参数与工具防御 | 未知工具、无效 schema、工具名/调用 ID 不匹配都会变成安全的结构化失败。 |
| 人工审批 | 写入前不会执行；批准/拒绝都产生可检查状态。 |
| 新进程恢复 | 新的 `ApplicationService` 仅靠 checkpoint 恢复并写入一次报告。 |
| 崩溃窗口 | 写入完成而结果 checkpoint 未保存时，恢复栅栏会拒绝重试，防止第二次副作用。 |
| 脱敏轨迹 | trace 不记录工具内容、参数正文或密钥。 |

重点入口：

- [`tests/test_integration.py`](tests/test_integration.py)：并发、失败隔离、审批和跨进程恢复的完整离线流程。
- [`tests/test_crash_recovery.py`](tests/test_crash_recovery.py)：真实临时文件与故障注入，证明恢复栅栏语义。

---

## 📁 迷你项目地图

```text
src/research_agent/
├── loop.py          # 项目自己拥有的多轮 Agent 控制流
├── registry.py      # 工具注册、schema 与参数验证
├── scheduler.py     # 异步并发、timeout、retry、失败隔离
├── state_store.py   # 原子 checkpoint、审批与恢复栅栏
├── trace.py         # 脱敏 JSONL 轨迹与摘要
├── runtime.py       # OpenAI-compatible 运行时组合根
└── tools/           # 文件与网页工具的安全能力边界

tests/                # 离线单元、集成与故障注入测试
docs/                 # 方案、运行记录与 README 插图
```

---

## 🧑‍⚖️ 给审阅者的 90 秒导游图

1. 看本页的“**一眼看懂运行方式**”，先理解控制权如何留在 Harness。
2. 打开 [`src/research_agent/loop.py`](src/research_agent/loop.py)，确认没有 Agent 框架接管控制循环。
3. 打开 [`src/research_agent/state_store.py`](src/research_agent/state_store.py)，查看 checkpoint、审批与执行栅栏。
4. 运行 `pytest -q`，再看 `test_integration.py` 与 `test_crash_recovery.py` 的可复验证据。
5. 任选一个 CLI demo，观察 `waiting_approval → completed` 及 `events.jsonl`。

---

## 🔍 这里有什么，也坦白没有什么

- 本项目刻意不包含向量数据库、多 Agent 编排、前端或复杂 RAG；它优先把单 Agent 的控制流、恢复语义和安全边界做完整。
- `fetch_url` 是受限资料获取工具，不是通用浏览器，也不承诺绕过站点策略。
- 系统提供的是“**副作用后崩溃时拒绝自动重复**”的保守语义，而不是对外宣称不可证明的 exactly-once。
- 真实模型只负责产出文本与工具提议；所有权限和状态迁移仍由本仓库代码决定。

更完整的设计说明见 [详细实现方案](docs/Agent_Harness_详细方案.md)，可复验的交付证据见 [运行记录](docs/运行记录.md)。

---

<p align="center">
  <sub>Built for inspectable, recoverable, human-governed agent execution. 🧡<br />
  模型可以天马行空，Harness 负责把安全带系好。</sub>
</p>
