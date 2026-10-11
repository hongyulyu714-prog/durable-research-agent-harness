# Durable Research Agent Harness

<p align="center">
  <img src="docs/images/durable-agent-hero.png" alt="研究者与可恢复研究 Agent 协作的插图" width="100%" />
</p>

<p align="center">
  <strong>一个从零实现、可恢复、可审计的研究型 Agent Harness。</strong><br />
  <sub>模型负责提出行动建议；控制流、权限边界、持久化与副作用管理由项目代码负责。</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/tests-44%20passed-22C55E" alt="44 tests passed" />
  <img src="https://img.shields.io/badge/Agent%20loop-project--owned-8B5CF6" alt="Project-owned agent loop" />
  <img src="https://img.shields.io/badge/Agent%20framework-none-F97316" alt="No agent framework" />
</p>

---

## 项目概述

`Durable Research Agent Harness` 是一个面向研究资料整理场景的轻量级 Agent 原型。它使用 OpenAI-compatible Chat Completions API 获取模型的文本和工具调用提议，但不依赖 LangChain、LangGraph、OpenAI Agents SDK 或其他 Agent 框架来控制执行流程。

项目关注的不是增加工具数量，而是回答 Agent 系统工程中更基础的问题：

- 模型提出工具调用后，谁负责校验、执行与回填结果？
- 多个独立工具如何并发运行，同时避免一个失败取消其他健康调用？
- 对文件写入等副作用，如何引入人工审批并支持跨进程恢复？
- 发生崩溃时，如何避免不安全地重复执行已经可能产生副作用的调用？
- 执行过程如何留下不暴露密钥和正文内容的可检查证据？

本仓库通过一个小而完整的控制平面实现这些能力：模型只能提出下一步；工具策略、状态迁移、审批和持久化均由 Harness 固定执行。

## 核心亮点

| 能力 | 实现位置 | 工程价值 |
|---|---|---|
| 自主多轮 Agent Loop | `src/research_agent/loop.py` | `AgentLoop` 自行驱动“模型提议 → 工具结果 → 下一轮模型 → 结束/暂停”的控制链，不依赖预编排 workflow。 |
| 异步并发与故障隔离 | `scheduler.py` | 独立只读工具可并发执行；超时、网络错误或参数错误会被转换为结构化 `ToolResult`，不会取消同批其他调用。 |
| Human-in-the-loop | `state_store.py` | 写入工具必须进入 `waiting_approval`；批准或拒绝后才恢复，且恢复时只执行已持久化的原始调用。 |
| 可恢复状态机 | `models.py`、`state_store.py` | RunState 可 JSON 序列化，checkpoint 原子写入；新服务实例可以仅依赖 `run_id` 加载并继续。 |
| 保守的副作用语义 | `ApprovalService` | 写入执行前落盘执行栅栏。若进程在写入完成、结果 checkpoint 保存前中断，恢复会拒绝盲目重试。 |
| 工具安全边界 | `registry.py`、`tools/` | Schema 校验、工作区路径边界、软链接防护和公开 URL 安全策略均不受模型文本控制。 |
| 可观测性与证据 | `trace.py`、`tests/` | 生成脱敏 JSONL 轨迹与摘要；关键语义由 44 项确定性离线测试覆盖。 |

> 项目不将“模型能调用工具”视为完成；重点在于让工具调用具备可控、可解释和可恢复的执行语义。

---

## 系统架构

<p align="center">
  <img src="docs/images/durable-agent-technical-roadmap.png" alt="命令行、模型、Agent 控制循环、工具校验、并发调度、人工审批、持久化与审计轨迹的技术路线图" width="100%" />
</p>

```text
                         User / CLI
                    run · resume · inspect
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Project-owned Harness                          │
│                                                                      │
│  AgentLoop ◄────────► LLM Adapter                                   │
│      │                    OpenAICompatibleLLM / MockLLM             │
│      │                                                               │
│      ├── ToolRegistry       schema、参数校验、静态工具策略            │
│      ├── AsyncToolScheduler 并发、超时、重试、失败隔离               │
│      ├── ApprovalService    审批、恢复、执行栅栏                     │
│      ├── StateStore         原子 checkpoint.json                     │
│      └── TraceRecorder      脱敏 events.jsonl / summary.json         │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                 read_file · fetch_url · write_report
```

### 组件职责

| 组件 | 职责 | 关键约束 |
|---|---|---|
| `OpenAICompatibleLLM` | 将兼容 API 的 Chat Completions 响应转换为项目内部 `LLMResponse`。 | 仅负责一次模型请求；不拥有 loop 或工具执行权。 |
| `AgentLoop` | 管理多轮状态、调用模型、处理工具结果、暂停审批和终止条件。 | 最大步数受限；未知工具或无效参数以模型可见失败结果回填。 |
| `ToolRegistry` | 保存工具定义、输出 OpenAI tool schema、验证模型来源参数。 | 只暴露已注册工具；参数必须通过严格的 JSON Schema 子集校验。 |
| `AsyncToolScheduler` | 执行有效的只读调用并收集结果。 | 仅并发 `parallel_safe` 调用；将超时/异常转换为结构化结果。 |
| `ApprovalService` | 对写入调用进行审批、拒绝、恢复和副作用保护。 | 只使用 checkpoint 中的待审批调用，不接受恢复时替换参数。 |
| `StateStore` | 保存与加载 `RunState`。 | 通过临时文件、`fsync`、原子替换保存 checkpoint。 |
| `TraceRecorder` | 输出可复核的事件和摘要。 | 对密钥及工具正文做脱敏或摘要化，避免泄露运行内容。 |

---

## 执行生命周期

<p align="center">
  <img src="docs/images/durable-agent-workflow.png" alt="Agent 从任务、工具执行、检查点、人工审批到报告和轨迹的执行流程" width="100%" />
</p>

### 状态机

| 状态 | 含义 | 可达后续状态 |
|---|---|---|
| `running` | Agent 正在请求模型、处理工具调用或继续执行。 | `waiting_approval`、`completed`、`failed` |
| `waiting_approval` | 已保存写入调用，正在等待人工批准或拒绝。 | `running`、`failed` |
| `completed` | 模型给出最终回答，或审批后的 continuation 已完成。 | 终态 |
| `failed` | 达到最大步数、模型响应无效或出现无法恢复的错误。 | 终态 |
| `cancelled` | 为运行状态预留的取消终态。 | 终态 |

### 正常执行路径

1. 用户通过 CLI 提交自然语言任务；系统创建 `RunState`。
2. `AgentLoop` 向模型发送已有消息和当前可用工具 schema。
3. 对只读调用，Harness 校验参数后由 scheduler 执行，并把 `ToolResult` 追加为 `role=tool` 消息。
4. 若模型提出 `write_report`，Harness 在执行前将完整调用写入 checkpoint，并把状态切换为 `waiting_approval`。
5. 用户在相同或新的进程中使用 `resume` 批准或拒绝该调用。
6. 批准时，`ApprovalService` 只执行保存的调用；随后 Agent 可以继续请求模型并产生最终回答。
7. 每个阶段都可在 `events.jsonl` 与 `summary.json` 中检查。

### 崩溃窗口与恢复策略

系统不声称无法证明的 exactly-once 语义。对于“文件已经写入，但写入结果尚未成功保存”这一窗口，恢复策略是：

1. 在执行写入工具前，将 `approval_execution_started=true` 持久化。
2. 如果之后发生中断，新进程加载 checkpoint 时仍能看到这个执行栅栏。
3. 再次 `resume --approve` 会抛出 `ApprovalRecoveryRequired`，而不是自动重复写入。
4. 操作者必须先检查报告目标，再决定如何处置。

该语义牺牲了自动重试的便利性，换取对重复副作用的保守保护。

---

## 内置工具与安全边界

| 工具 | 能力范围 | 风险等级 | 安全控制 |
|---|---|---:|---|
| `read_file(path)` | 读取工作区 `data/` 下的 UTF-8 文本。 | `read` | 路径必须位于 `data/`；路径逃逸和指向工作区外的软链接会被拒绝。 |
| `fetch_url(url)` | 异步获取公开 HTTP(S) 页面中的受限文本。 | `read` | 拒绝 localhost、私网、metadata 地址和不安全重定向；限制响应体大小。 |
| `write_report(path, content)` | 写入工作区 `reports/` 下已存在的目录。 | `write` | 需要人工审批；普通 scheduler 无法直接执行；恢复时使用保存的调用参数。 |

`ToolSpec` 是策略信任边界。模型可以提出调用，但不能通过提示词、参数或工具名称改变下列事项：

- 工具是否存在；
- 参数 schema 与允许字段；
- 文件系统访问范围；
- 是否允许并发、超时与重试策略；
- 是否需要人工审批。

---

## 安装与配置

### 环境要求

- Python 3.11 或更高版本
- 一个兼容 Chat Completions 的模型服务
- 用于真实运行的 `OPENAI_API_KEY` 与 `OPENAI_MODEL`

### 安装依赖

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 配置模型服务

```bash
cp .env.example .env
```

编辑 `.env`：

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=your_model_name
```

加载变量：

```bash
set -a; source .env; set +a
```

`.env` 已被 Git 忽略。请勿提交密钥、真实任务正文、未经审核的网页资料或运行产生的 checkpoint。

---

## 快速开始

### 1. 准备工作区

工作区将数据读取和报告写入分隔到两个显式目录：

```bash
mkdir -p demo-workspace/data demo-workspace/reports demo-runs
printf 'Durable execution makes research workflows inspectable.\n' \
  > demo-workspace/data/source.txt
```

### 2. 启动任务

```bash
research-agent --runs-dir demo-runs run \
  "Read source.txt, check one public source, then write a concise Markdown brief." \
  --workspace demo-workspace
```

如果模型提出写入报告，命令会输出：

```text
waiting_approval: <run_id>
```

此时 `demo-runs/<run_id>/checkpoint.json` 已保存待审批调用；报告尚未写入。

### 3. 审批、拒绝或检查

批准写入：

```bash
research-agent --runs-dir demo-runs resume <run_id> --approve \
  --workspace demo-workspace
```

拒绝写入并提供原因：

```bash
research-agent --runs-dir demo-runs resume <run_id> \
  --reject "请先缩短为三段" --workspace demo-workspace
```

检查任意运行摘要（不需要 API Key）：

```bash
research-agent --runs-dir demo-runs inspect <run_id>
```

### CLI 命令参考

| 命令 | 用途 | 是否要求模型凭据 |
|---|---|---:|
| `research-agent run TASK --workspace PATH` | 创建并执行一次新任务。 | 是 |
| `research-agent resume RUN_ID --approve --workspace PATH` | 批准并恢复待审批任务。 | 取决于后续是否需要模型继续执行。 |
| `research-agent resume RUN_ID --reject REASON --workspace PATH` | 拒绝调用，并将原因作为工具结果回填。 | 取决于后续是否需要模型继续执行。 |
| `research-agent inspect RUN_ID` | 输出本地运行摘要。 | 否 |

---

## 运行记录与可观测性

每个 run 保存在 `runs/<run_id>/` 下：

| 文件 | 内容 | 用途 |
|---|---|---|
| `checkpoint.json` | 完整、可序列化的 `RunState`。 | 恢复运行、检查待审批调用和状态机。 |
| `events.jsonl` | 追加式、脱敏的事件流。 | 审计 loop、工具批次、审批与恢复路径。 |
| `summary.json` | 事件数、工具失败数、审批决策等概要。 | 快速检查运行结论。 |

轨迹记录避免直接写入工具正文、请求参数正文和密钥：敏感字段会被替换为 `[REDACTED]`，内容字段仅保留长度和短哈希。这样既支持验证执行路径，也尽量降低日志泄露风险。

---

## 测试与验证

完整测试集离线、确定性执行，不需要 API Key：

```bash
PYTHONPATH=src python3.11 -m pytest -q
python3.11 -m ruff check src tests
python3.11 -m compileall -q src tests
```

当前结果：**44 passed**。

| 验证类别 | 代表性覆盖 |
|---|---|
| Agent Loop | 多轮消息、工具调用回填、无效工具处理、最大步数终止。 |
| 工具注册与策略 | schema 验证、工具 ID/名称对应、读写风险策略、路径与软链接边界。 |
| 异步调度 | 并发执行、超时、重试、网络异常与失败隔离。 |
| 审批与恢复 | 暂停、批准、拒绝、多写入顺序，以及新服务实例恢复。 |
| 故障注入 | 真实临时报告写入后模拟 checkpoint 失败，验证恢复栅栏阻止重复执行。 |
| 轨迹与摘要 | JSONL 脱敏、事件去重、状态摘要与运行证据。 |

重点测试入口：

- [`tests/test_integration.py`](tests/test_integration.py)：并发读取、网页失败隔离、审批暂停、跨服务恢复、轨迹与摘要。
- [`tests/test_crash_recovery.py`](tests/test_crash_recovery.py)：真实文件写入与崩溃窗口的故障注入证明。

---

## 项目结构

```text
src/research_agent/
├── cli.py           # run / resume / inspect 命令行入口
├── llm.py           # OpenAI-compatible 与离线 MockLLM 适配器
├── loop.py          # 项目拥有的多轮 Agent 控制流
├── models.py        # RunState、状态机、工具调用与结果契约
├── registry.py      # 工具注册、schema 与参数验证
├── runtime.py       # 生产组合根与环境变量配置
├── scheduler.py     # 异步并发、timeout、retry、失败隔离
├── state_store.py   # 原子 checkpoint、审批与恢复栅栏
├── trace.py         # 脱敏 JSONL 轨迹与摘要
└── tools/           # 文件与网页工具及其能力边界

tests/                # 单元、集成与故障注入测试
docs/                 # 详细方案、运行记录、架构插图
examples/             # 可参考的任务与工作区示例
```

---

## 设计边界与后续扩展

本项目有意将范围控制在单 Agent、三个窄工具和可验证的执行语义。它不包含向量数据库、多 Agent 编排、通用浏览器自动化或前端界面；这些能力可在稳定的控制平面之上继续扩展。

当前明确的设计边界包括：

- `fetch_url` 是受限资料获取工具，不是绕过站点策略的通用浏览器；
- 写入路径必须事先创建，工具不会隐式扩大文件系统能力；
- 副作用后的崩溃窗口采用“阻止自动重复、要求人工检查”的保守恢复策略；
- 模型服务仅用于生成文本与工具提议，不参与权限判断或状态迁移。

## 参考文档

- [详细实现方案](docs/Agent_Harness_详细方案.md)
- [运行记录与复核说明](docs/运行记录.md)
- [项目 GitHub 仓库](https://github.com/hongyulyu714-prog/durable-research-agent-harness)
