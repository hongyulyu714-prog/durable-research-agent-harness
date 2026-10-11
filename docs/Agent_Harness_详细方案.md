# Durable Research Agent Harness：详细实现方案

## 1. 项目定位

本项目实现一个轻量、可运行的研究型 Agent / Harness 原型。用户以自然语言提交研究任务；模型可自主选择工具、根据工具结果继续推理，并最终给出结论或请求写入研究简报。

项目的重点不是工具数量，而是由本项目代码掌握完整运行控制：**多轮 Agent loop、并发工具调度、失败处理、Human-in-the-loop（HITL）、可恢复状态与可检查轨迹**。

### 1.1 演示场景

用户输入：

> 阅读 `data/` 中的本地材料，并获取指定的两个公开网页；基于可用资料撰写 Markdown 研究简报，保存到 `reports/research_brief.md`。

预期行为：

1. LLM 自主选择 `read_file`、`fetch_url` 等工具；不是由 Python 预先写死步骤。
2. 同一轮内的独立只读工具并发运行。
3. 单个网页超时或文件不存在时，成功工具的结果仍会回填；失败以结构化错误反馈给 LLM，由 LLM 决定调整方案或继续。
4. 模型提出 `write_report` 时，Harness 根据策略暂停运行、持久化状态，等待人工批准或拒绝。
5. 通过新的 CLI 进程批准后，系统从 checkpoint 恢复，只执行此前未完成的待审批调用，再继续模型循环直至完成。

## 2. 题目要求与实现对应

| 题目要求 | 方案中的实现与证据 |
|---|---|
| 自行实现 Agent 控制流 | `AgentLoop.run()` 自己维护 `while` 循环、最大步数、模型调用、工具调用、暂停与终止条件。 |
| 支持工具使用 | `ToolRegistry` 提供 JSON Schema、参数验证、工具分派、结构化成功/失败结果。 |
| 真实异步 | `AsyncToolScheduler` 用 `httpx.AsyncClient` 执行网络 I/O，并以 `asyncio.gather`、`Semaphore`、timeout 并行调度同轮安全调用。 |
| Human-in-the-loop | 高风险工具进入 `WAITING_APPROVAL`，将完整状态与待审批调用写入 checkpoint；`resume` 命令带审批决定恢复。 |
| 可运行、可检查 | pytest、离线 MockLLM、真实 API demo、JSONL trace、运行摘要和 README。 |

## 3. 明确的边界与取舍

### 3.1 使用的依赖

| 依赖 | 用途 | 是否参与核心 Agent 控制链 |
|---|---|---|
| `openai` | 基础 OpenAI-compatible Chat Completions API 客户端 | 否；仅发送模型请求与读取响应。 |
| `httpx` | `fetch_url` 的异步 HTTP I/O | 否。 |
| `pydantic`（可选） | 工具参数、状态文件的输入校验 | 否。 |
| `pytest` | 自动化测试 | 否。 |

不使用 LangChain、LangGraph、OpenAI Agents SDK、DeepAgents 或其他替项目实现 Agent loop、工具编排或状态机的框架。`asyncio`、`json`、`pathlib`、`uuid`、`time` 等均使用 Python 标准库。

### 3.2 不实现的能力

- 不做 Web UI；CLI 足以完整展示暂停与恢复。
- 不做多 Agent、向量数据库、浏览器自动化、长期记忆或分布式队列。
- 不承诺生产级 exactly-once：本地文件系统仍有进程在副作用完成但状态未落盘的崩溃窗口。对此以稳定调用 ID、原子状态写入和保守重试来降低风险，并在 README 如实说明。

## 4. 总体架构

```text
CLI (run / resume / inspect)
            |
            v
        AgentLoop  <---------------------------+
            |                                  |
            v                                  |
     OpenAICompatibleLLM                        |
            |                                  |
      0..N ToolCall                             |
            v                                  |
    PolicyEngine ---- approval required ----> StateStore
            |                                      |
            | allow                                | checkpoint JSON
            v                                      v
    AsyncToolScheduler <------------------- HITL resume
       |          |
       |          +-- Semaphore / timeout / retry
       v
  ToolRegistry (read_file, fetch_url, write_report)
       |
       v
  structured ToolResult -> messages -> next LLM turn

TraceRecorder 记录上述每个状态变化、LLM、工具、审批事件到 JSONL。
```

## 5. 模块划分

```text
research_agent/
├── __main__.py          # python -m research_agent
├── cli.py               # run、resume、inspect 命令
├── loop.py              # 自实现 Agent 控制循环
├── llm.py               # OpenAI-compatible adapter 与 MockLLM
├── models.py            # RunState、ToolCall、ToolResult、Event 等数据模型
├── registry.py          # Tool 定义、Schema、参数校验与查询
├── scheduler.py         # 并发、timeout、重试、失败隔离
├── policy.py            # 风险等级、审批策略、路径限制
├── state_store.py       # 原子 checkpoint 的保存与加载
├── trace.py             # JSONL 事件记录与摘要
└── tools/
    ├── files.py         # read_file、write_report
    └── web.py           # fetch_url
tests/
examples/
data/
reports/
runs/                    # 运行时生成；每个 run 一个目录
```

每个运行目录为：

```text
runs/<run_id>/
├── checkpoint.json      # 当前可恢复状态
├── events.jsonl         # 追加式执行轨迹
└── summary.json         # 完成后生成的统计摘要
```

## 6. 核心数据结构

### 6.1 工具元数据

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters_schema: dict
    risk: Literal["read", "write"]
    parallel_safe: bool
    timeout_seconds: float
    max_retries: int
```

关键原则：模型只能提出工具调用；是否并发、是否重试、是否审批均由 Harness 的 `ToolSpec` 和 `PolicyEngine` 决定。

初版工具如下：

| 工具 | 作用 | 并发 | 审批 | 重试 |
|---|---|---:|---:|---:|
| `read_file(path)` | 读取 `data/` 中 UTF-8 文本 | 是 | 否 | 0 |
| `fetch_url(url)` | 获取允许协议的公开 URL 文本 | 是 | 否 | 2（瞬态错误） |
| `write_report(path, content)` | 写入 `reports/` 下 Markdown 报告 | 否 | 是 | 0 |

### 6.2 统一工具结果

无论调用成功、参数错误、超时、未知工具还是审批拒绝，均形成可 JSON 序列化的 `ToolResult`：

```json
{
  "tool_call_id": "call_a1b2",
  "tool_name": "fetch_url",
  "ok": false,
  "data": null,
  "error": {
    "kind": "timeout",
    "message": "fetch_url exceeded 8.0 seconds",
    "retryable": true
  },
  "attempts": 2,
  "duration_ms": 8017
}
```

工具异常不会直接终止 Agent loop。Harness 捕获预期异常并以工具消息回填，让模型获取失败事实后重新规划。`CancelledError` 始终向上传播，避免误把取消错误当作正常失败。

### 6.3 可恢复运行状态

```python
class RunStatus(str, Enum):
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class RunState:
    run_id: str
    status: RunStatus
    task: str
    messages: list[dict]
    step: int
    completed_call_ids: set[str]
    pending_calls: list[ToolCall]
    pending_approval: ToolCall | None
    final_answer: str | None
```

`pending_approval` 保存的是模型本轮提出的原始工具名、参数、`tool_call_id` 和参数摘要。恢复时必须执行此快照，不重新请求模型来生成另一个写入操作；如果参数不同，就是新的工具调用，必须再次审批。

## 7. Agent loop 与状态机

### 7.1 状态转换

```text
NEW -> RUNNING -> COMPLETED
             ├-> FAILED
             └-> WAITING_APPROVAL -> RUNNING
                                  └-> CANCELLED
```

合法状态迁移由 `transition(state, event)` 集中校验；非法迁移，例如 `COMPLETED -> RUNNING`，直接拒绝并记录错误事件。

### 7.2 伪代码

```python
async def run(state: RunState) -> RunState:
    while state.status is RUNNING:
        assert state.step < MAX_STEPS
        response = await llm.complete(state.messages, registry.schemas())
        trace.record("llm_response", step=state.step)

        if response.is_final:
            state.messages.append(response.assistant_message)
            state.final_answer = response.text
            state.status = COMPLETED
            store.save(state)
            break

        calls = parse_and_validate(response.tool_calls)
        state.messages.append(response.assistant_message)
        decision = policy.classify(calls)

        if decision.requires_approval:
            state.pending_approval = decision.call
            state.status = WAITING_APPROVAL
            store.save(state)              # 先持久化，再向 CLI 返回暂停
            trace.record("approval_requested", call_id=decision.call.id)
            break

        results = await scheduler.execute(calls)
        append_tool_results(state.messages, results)
        state.completed_call_ids.update(result.tool_call_id for result in results)
        state.step += 1
        store.save(state)

    return state
```

恢复时：先加载 checkpoint、校验当前状态只能为 `WAITING_APPROVAL`；记录 `approval_granted` 或 `approval_rejected`。批准时仅执行保存的那一个 `pending_approval`，拒绝时生成 `kind="approval_rejected"` 的 ToolResult。两种结果都会写入 messages，随后状态恢复为 `RUNNING` 并继续 loop。

## 8. 异步调度策略

### 8.1 真实并发

模型可能在同一响应中提出三个独立的 `fetch_url`/`read_file` 调用。调度器会：

1. 只挑选 `parallel_safe=True`、且无审批要求的调用组成并发批次；
2. 以 `asyncio.Semaphore(max_concurrency=3)` 限制外部 I/O；
3. 每个调用在自己的 `asyncio.timeout()` 作用域运行；
4. 以安全包装函数将异常转成 ToolResult；
5. 使用 `asyncio.gather()` 收集全体结果，而不是让一个失败取消其他调用；
6. 按原始 tool call 顺序回填结果，确保模型消息协议稳定。

不能直接使用未包装的 `TaskGroup` 来实现本场景：任一子任务抛出异常时，它会取消仍在运行的子任务。该语义适合“全成或全败”的任务组，却不符合本项目要求的“失败隔离”。

### 8.2 重试规则

| 错误类型 | 自动重试 | 理由 |
|---|---:|---|
| 网络超时、临时 5xx、连接错误 | 是，最多 2 次、指数退避 | 多数是短暂依赖失败。 |
| 参数校验、路径越界、404 | 否 | 重试不会改变结果。 |
| 审批拒绝、策略拒绝 | 否 | 不允许自动绕过人或策略。 |
| `write_report` | 否 | 副作用工具不可盲目重复。 |

所有重试尝试都写入 trace；最终失败仍作为结构化 ToolResult 交回模型。

## 9. Human-in-the-loop 与安全策略

### 9.1 风险分级

```text
模型提出调用
      |
      v
参数与路径校验 --失败--> 结构化错误回填模型
      |
      v
PolicyEngine
  | read             | write
  v                  v
自动执行          保存 checkpoint，等待人工决定
```

- `read_file` 只能访问 `data/`；`write_report` 只能访问 `reports/`。
- 通过 `Path.resolve()` 与 `relative_to(allowed_root)` 防止 `..`、绝对路径或软链接绕过。
- 高风险写入默认人工批准；审批永远由 Harness 决定，模型不能通过文字声明“该操作安全”绕过。
- 来自网页和文件的内容被包装为标记的**不可信工具数据**，系统提示明确要求将其视为资料而非指令。

### 9.2 审批 CLI

```bash
python -m research_agent run "阅读 data 中资料并输出报告"
# 输出：WAITING_APPROVAL，显示 run_id、工具名、目标路径、内容摘要

python -m research_agent resume <run_id> --approve
python -m research_agent resume <run_id> --reject "请改为更短的报告"
python -m research_agent inspect <run_id>
```

审批界面不显示 API Key 等敏感环境变量；写入内容仅展示截断摘要和 SHA-256 摘要，完整参数只保存在本地 checkpoint。

## 10. 可观测性、回放与验证

### 10.1 事件轨迹

每行一个 JSON event，追加写入并立即 flush：

```json
{"ts":"...","run_id":"run_01","step":1,"type":"tool_started","call_id":"c1","tool":"fetch_url"}
{"ts":"...","run_id":"run_01","step":1,"type":"tool_finished","call_id":"c1","ok":true,"duration_ms":823}
{"ts":"...","run_id":"run_01","step":2,"type":"approval_requested","tool":"write_report"}
```

`inspect` 命令生成简短时间线，显示状态、轮数、每个工具时长、并发峰值、失败次数与审批决定。trace 不记录完整网页原文和 API Key；工具参数只记录必要摘要。

### 10.2 录制与回放

提供 `MockLLM`，以预先定义的模型响应序列驱动 loop。这样无需 API Key 或网络也可验证：模型调用 → 工具回填 → 下一轮模型调用的完整协议。

测试不追求宣称“真实模型输出确定”；真实 API run 只作为运行证明。离线测试固定模型响应并断言状态和轨迹，因此具有确定性、零成本与可重复性。

## 11. 测试计划

| 测试 | 证据 |
|---|---|
| `test_loop_multi_turn` | MockLLM 先请求工具、后给最终答案；断言至少两次模型调用且 tool result 在第二次输入中。 |
| `test_unknown_or_invalid_tool` | 不存在工具/非法 JSON 参数转为结构化错误，loop 不崩溃。 |
| `test_parallel_speed_and_overlap` | 三个可控延迟的 async 工具总时长接近最长单项时长，而不是延迟之和；trace 有重叠 started/finished 区间。 |
| `test_failure_isolation` | 同批一个工具抛超时，其他工具仍成功并回填。 |
| `test_retry_policy` | 瞬态失败恰好重试两次；校验错误、拒绝和写入不重试。 |
| `test_waiting_approval_never_writes` | 运行至审批前，目标文件不存在且 checkpoint 为 `WAITING_APPROVAL`。 |
| `test_resume_approve_once` | 新建对象加载 checkpoint 后只执行 pending 的写入调用，不重复已完成读取。 |
| `test_resume_reject` | 拒绝生成 `approval_rejected` ToolResult 并令模型继续决策。 |
| `test_path_policy` | `../`、绝对路径和软链接越界均被拒绝。 |
| `test_step_budget` | 超过最大轮数转为明确失败状态，防止无限循环。 |

## 12. Demo 与提交物

### 12.1 Demo 流程

1. 运行真实 OpenAI-compatible API 的演示命令。
2. trace 显示本地读取与两个网页获取在同一轮并发开始；其中一个故意指向慢响应/错误 URL，展示失败隔离。
3. 第二轮模型依据成功资料提出 `write_report`；系统暂停，并展示 checkpoint 路径。
4. 在新命令中运行 `resume --approve`；系统写入报告、回填工具结果并完成。
5. 提供上述 run 的 `events.jsonl`、`checkpoint.json`（脱敏）和 `summary.json`。
6. 执行 `pytest -q`，保存测试输出。

### 12.2 README 必须包含

- 安装、环境变量与模型配置；
- `run`、`resume`、`inspect`、`pytest` 命令；
- 架构图与 Agent loop 控制权说明；
- 工具表、并发条件、审批策略与错误语义；
- 依赖清单与“未使用 Agent/Harness 框架”的边界说明；
- 一次真实运行记录以及限制说明；
- GitHub 调研参考和许可证/归属说明。

## 13. GitHub 调研的具体借鉴与限制

| 参考 | 借鉴内容 | 在本项目中的处理 |
|---|---|---|
| OpenAI Cookbook Function Calling | 由应用检查 tool call、执行函数、以相同 call ID 回填、再次请求模型 | 使用基础 `openai` SDK，但 loop 与分派自行实现。 |
| `aafre/agentharness` | 将副作用记录为 event、状态与记录可检查/回放 | 不安装该包；自写简化的 JSONL event 与 RunState。 |
| `nderman/agent-harness` | 轨迹断言、预录模型响应、故障注入测试 | 不复用 Agent runtime；采用 MockLLM 和 pytest fixture。 |
| Temporal Agent Harness 示例 | 副作用工具使用稳定调用 ID；预期失败应作为数据返回 LLM | 不使用 Temporal；本地 JSON checkpoint 和保守写入策略实现原型。 |
| Agent Prompt Shield | 将网页/工具输出视为不可信数据，策略门控敏感操作 | 不引入检测框架；仅实现明确的数据标记、路径限制与审批门。 |

任何直接复制的非核心代码都必须在实现前确认上游许可证兼容性、保留 required notice，并在 README 的 Attribution 小节说明。优先选择自己实现短小且可解释的版本；本题的评分重点是项目自身 Harness 的控制与工程判断。

## 14. 6 小时实施优先级

| 优先级 | 内容 | 目标时间 |
|---:|---|---:|
| P0 | 项目初始化、数据模型、MockLLM、基础多轮 loop | 60 分钟 |
| P0 | 三个工具、注册表、参数校验、结构化错误 | 45 分钟 |
| P0 | `asyncio` 并发、timeout、失败隔离与相关测试 | 45 分钟 |
| P0 | checkpoint、审批/拒绝/恢复与幂等保护 | 75 分钟 |
| P1 | JSONL trace、`inspect` 摘要、路径安全 | 30 分钟 |
| P0 | pytest、真实 API demo、README 与运行记录 | 65 分钟 |
| P2 | 轻量工具输出 injection 关键字告警、回放 CLI | 20 分钟 |

若时间不足，绝不牺牲 P0 来换取界面、更多工具或多 Agent。完成 P0 已能严谨覆盖题目全部硬性要求，并以可恢复审批、失败隔离并发和离线轨迹测试形成差异化亮点。

## 15. 验收标准

提交前逐项确认：

- [ ] 不使用任何替代项目来执行 Agent loop、工具调度或状态机。
- [ ] LLM 至少在真实 demo 中完成一次“工具调用 → 结果回填 → 下一轮模型调用”。
- [ ] 同一模型响应中的独立 I/O 工具发生真实并发，且有耗时/trace 证据。
- [ ] 一个工具失败不会取消其余独立工具，错误会回到 LLM。
- [ ] `write_report` 在批准前绝不执行；重启后可以批准或拒绝并恢复。
- [ ] 每次运行保留状态与事件轨迹；pytest 覆盖核心状态转移。
- [ ] README 让第三方可配置、运行、复核 demo 与理解边界。

