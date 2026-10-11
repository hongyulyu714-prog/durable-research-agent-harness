# Durable Research Agent Harness

一个从零实现的、可恢复的研究型 Agent 原型。模型提出工具调用，但多轮控制循环、并发调度、参数校验、人工审批、checkpoint 与运行轨迹均由本项目代码掌握；不使用 LangChain、LangGraph、OpenAI Agents SDK 或其他 Agent/Harness 框架。

## 能力与边界

- `read_file(path)`：只读取工作区 `data/` 下的 UTF-8 文本。
- `fetch_url(url)`：异步读取公开 HTTP(S) 页面；拦截本地、私网、metadata 地址与不安全重定向，并限制响应大小。
- `write_report(path, content)`：只写入工作区 `reports/`；必须经人工批准，普通调度器不会执行它。
- 同一模型回合内，独立的只读工具并发执行。单个超时、网络错误或参数错误会变为结构化 `ToolResult`，不取消健康的同级调用。
- 批准前完整调用会原子保存至 `runs/<run_id>/checkpoint.json`；`resume` 只执行这份保存的调用。
- `events.jsonl` 和 `summary.json` 提供脱敏、可检查的轨迹。工具内容、参数正文与密钥不写入轨迹。

这不是 production-grade exactly-once 系统：若进程在写入完成、结果 checkpoint 落盘前崩溃，恢复会拒绝盲目重试，要求先人工检查目标文件。

## 安装与配置

需要 Python 3.11 或更高版本。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
set -a; source .env; set +a
```

`.env` 需要 `OPENAI_API_KEY` 与 `OPENAI_MODEL`；`OPENAI_BASE_URL` 默认为 OpenAI API 地址。请勿提交 `.env`、含私密任务正文的 checkpoint 或未审核的网页资料。

## 运行、审批与检查

准备一个工具工作区：

```bash
mkdir -p demo-workspace/data demo-workspace/reports demo-runs
printf 'Durable execution makes research workflows inspectable.\n' > demo-workspace/data/source.txt
```

```bash
research-agent --runs-dir demo-runs run \
  "Read source.txt, check one public source, then write a concise Markdown brief." \
  --workspace demo-workspace
```

当输出为 `waiting_approval: <run_id>` 时，可在新进程中批准已持久化的写入：

```bash
research-agent --runs-dir demo-runs resume <run_id> --approve --workspace demo-workspace
```

或拒绝它，并将原因作为结构化工具结果回填给模型：

```bash
research-agent --runs-dir demo-runs resume <run_id> --reject "请先缩短为三段" --workspace demo-workspace
```

`inspect` 不需要 API 凭据：

```bash
research-agent --runs-dir demo-runs inspect <run_id>
```

## 离线验证

```bash
pytest -q
ruff check .
```

`tests/test_integration.py` 是完整的离线验收路径。它使用 `MockLLM` 与受控异步工具，断言本地读取和网页调用实际重叠、网页故障不影响读取结果、写入在批准前不会执行、新的 `ApplicationService` 仅凭 checkpoint 恢复，以及 JSONL/summary 包含完整的 loop、批次、故障、审批与完成轨迹。测试不访问网络，也不需要 API Key。

`tests/test_crash_recovery.py` 则是针对持久化语义的故障注入证明。它以真实的临时工作区文件验证两条路径：新的服务实例可仅凭 checkpoint 在批准后写入一次报告；若报告已写入但结果 checkpoint 尚未保存时模拟崩溃，新的服务实例会看到持久化的执行栅栏、拒绝盲目重试，并要求人工先检查目标文件。该设计避免把“至多一次”误表述为无法证明的 exactly-once 保证。

## 架构

```text
CLI (run / resume / inspect)
        |
        v
AgentLoop -- OpenAICompatibleLLM / MockLLM
        |
        +-- ToolRegistry -- read_file / fetch_url / write_report
        |        |
        |        +-- AsyncToolScheduler (并发、timeout、retry、失败隔离)
        |
        +-- ApprovalService -- StateStore (原子 checkpoint)
        |
        +-- TraceRecorder (脱敏 JSONL 与 summary)
```

`ToolSpec` 是策略信任边界：模型的文字不能改变写入审批、并发、重试或路径范围。每个工具批次会记录总耗时与所有单工具耗时之和；当前者更小时，可从轨迹核验独立调用发生了重叠。

## 依赖

| 依赖 | 用途 | 是否掌握 Agent 控制链 |
|---|---|---:|
| `openai` | 基础 OpenAI-compatible Chat Completions 客户端 | 否 |
| `httpx` | `fetch_url` 的异步 HTTP I/O | 否 |
| `pydantic` | 可选的数据模型/输入校验依赖 | 否 |
| `pytest`、`pytest-asyncio`、`ruff` | 开发期离线验证与检查 | 否 |

完整的设计取舍见 [详细实现方案](docs/Agent_Harness_详细方案.md)，可复验的交付证据见 [运行记录](docs/运行记录.md)。
