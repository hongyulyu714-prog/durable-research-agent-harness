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

## 13. 高 star GitHub 调研：可借鉴模式与禁止边界

以下 star 数由 GitHub API 于 **2026-10-11** 核验。选择这些项目不是为了将成熟框架包装后提交，而是为了从经大规模社区使用的设计中抽取最小、可解释的工程原则。题目明确禁止它们承担本项目的 Agent loop、核心工具调度或 Harness 状态编排。

| 项目 | GitHub star（核验日） | 成熟设计中值得吸收的点 | 本项目的自实现落点 | 为什么不直接使用 |
|---|---:|---|---|---|
| [LangChain](https://github.com/langchain-ai/langchain) | 147,585 | 工具定义、模型适配和中间件分层的工程边界 | `llm.py`、`registry.py`、`policy.py` 维持清晰接口，避免把工具逻辑写进 CLI 或 prompt | LangChain 不得承担本项目的循环、调度或状态管理。 |
| [OpenAI Cookbook](https://github.com/openai/openai-cookbook) | 76,548 | Function Calling 的基本协议：模型只提出调用；应用校验、执行，并以相同 call ID 回填结果后继续请求模型 | `AgentLoop` 显式维护 assistant tool-call message 与 role=`tool` 结果消息；无隐式自动执行 | Cookbook 是示例集合而不是本项目运行时框架；只使用基础 `openai` 客户端。 |
| [Microsoft AutoGen](https://github.com/microsoft/autogen) | 61,344 | `save_state`/`load_state` 将**配置**与**运行时状态**分离；重启后从外置状态恢复 | `RunState` 只保存消息、步骤、待审批调用和已完成调用；工具/模型配置由新进程重新构建 | AutoGen 本身是题目所排除的 Agent framework，绝不作为依赖。 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | 43,045 | interrupt + checkpoint 的 HITL 模式；其文档特别提示中断后节点可能重新执行，因而副作用需可幂等 | 审批前落盘 `pending_approval`；恢复时只执行该快照调用；写操作不用自动重试，并使用稳定 call ID 去重 | LangGraph 不能接管本项目状态机；本项目用 JSON checkpoint 和自写状态迁移。 |
| [openai-python](https://github.com/openai/openai-python) | 31,806 | `AsyncOpenAI` 与基础 API 的异步调用能力 | `llm.py` 只封装 `AsyncOpenAI`；项目自己决定何时下一轮、执行哪些工具或暂停 | 这是允许的基础 API SDK；它不拥有 Agent loop。 |
| [Semantic Kernel](https://github.com/microsoft/semantic-kernel) | 28,640 | 函数调用前的 policy/filter 可阻止执行并返回拒绝结果，而不是让模型自行绕过 | `PolicyEngine` 在工具真正运行前返回 allow / require-approval / deny，并将拒绝包装为 ToolResult | Semantic Kernel 不进入依赖树，避免其自动函数调用能力接管控制链。 |
| [Temporal](https://github.com/temporalio/temporal) | 23,594 | Durable execution 的三个关键概念：持久化检查点、长时间等待外部 signal、带幂等键的副作用 | `checkpoint.json`、CLI `resume` 和稳定 `tool_call_id` 提供 Prototype 级别的等价展示 | Temporal 是超出 6 小时范围的外部工作流平台；不需要部署服务。 |

### 13.1 从高 star 项目提炼出的四个“可见亮点”

#### 亮点 A：持久化审批恢复，而不是终端 `input()`

LangGraph 与 AutoGen 都将“暂停后可恢复”视为运行时状态问题，而不是 UI 问题。本项目的差异化做法是将审批请求表示为**不可变的、带 call ID 的待执行快照**：

```text
LLM 提出 write_report(call_id=c42, path=..., content=...)
    -> PolicyEngine 标记需审批
    -> checkpoint.json 写入 c42 的原始参数
    -> 新进程 resume --approve
    -> 只执行 c42，且不重新请求 LLM 生成另一个写入
```

这既可展示跨进程恢复，也避开“恢复时重新执行整轮、导致重复副作用”的常见风险。测试将明确验证：审批前无文件；批准后只出现一次写入；拒绝生成 LLM 可见的 `approval_rejected` 结果。

#### 亮点 B：并发但不牺牲失败隔离

高层 Agent 框架常默认隐藏工具并发；本项目刻意将其变成可观测的 Harness 行为。`AsyncToolScheduler` 按工具元数据而不是模型的自我声明决定策略：

- 只读、独立的 `read_file`/`fetch_url` 用 `asyncio.gather` 并发；
- `Semaphore` 控制最大外部请求数；
- 每个任务独立 timeout 和重试包装，异常变成 ToolResult；
- 写入调用不进入并发批次，且必须先过审批；
- JSONL trace 记录每个工具 started/finished 时间，测试用可控延迟证明总耗时接近最长单项而非所有项之和。

这里刻意不直接使用未包装的 `TaskGroup`：其异常语义会取消同组的其他任务，违背“一个网页失败、其余资料仍可被模型使用”的研究任务需求。

#### 亮点 C：轨迹级离线回归，而非只检查最终文案

从高 star 项目之外的专门 Harness 实践中补充吸收“cassette / golden trajectory”测试模式。每个 `MockLLM` 场景预置模型响应，测试严格断言：

```text
model response #1 -> 工具 c1/c2/c3 -> 结构化结果回填 -> model response #2
                                              -> approval requested -> resume -> final
```

这比“最终 Markdown 中含某个字符串”更能证明 Agent loop、错误传递和 HITL 状态语义真的正确，也避免真实模型的随机性令 CI 不稳定。

#### 亮点 D：模型无权覆盖 Harness 策略

借鉴 Semantic Kernel 的 function-approval 过滤思想，模型只能**提议**工具调用；真正的权限判断由项目代码完成。`ToolSpec` 固定定义 `risk`、`parallel_safe`、timeout 和 retry；`PolicyEngine` 固定定义允许目录与审批规则。即使外部网页诱导模型执行写入，受保护工具也不会因为模型文本而跳过策略。

### 13.2 精简参考项目：直接对应的实现细节

| 参考 | 使用场景 | 吸收的具体细节 |
|---|---|---|
| [OpenAI Cookbook：Function Calling](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb) | 原生工具调用协议 | 校验 `tool_calls`、解析 arguments、保持 `tool_call_id` 对应关系、回填 tool message 后再次调用模型。 |
| [AutoGen 状态持久化示例](https://github.com/microsoft/autogen/tree/main/python/samples/agentchat_fastapi) | checkpoint 数据范围 | 状态文件用于恢复运行，配置文件用于重新构建实例；二者不混淆。 |
| [LangGraph interrupt 源码注释](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py) | 恢复与幂等 | 中断后可能从节点开头重跑，因此任何写入工具都必须避免盲目重试；这是本项目稳定 call ID 的设计依据。 |
| [Semantic Kernel 审批样例](https://github.com/microsoft/semantic-kernel/blob/main/dotnet/samples/Demos/FunctionInvocationApproval/Program.cs) | 策略拒绝语义 | 在工具执行前作决定；拒绝返回明确结果，而不是抛出未处理异常。 |
| [Temporal Agent Harness 示例](https://github.com/mikeacjones/temporal-agent-harness-example) | 副作用与失败语义 | 读取类工具与写入类工具采用不同重试策略；预期异常转为 LLM 可见结果。 |
| [Agent Prompt Shield](https://github.com/0xMrBlueOps/agent-prompt-shield) | 不可信工具输出边界 | 工具输出标记为数据、保留来源，并在高权限工具前执行策略门控。 |

### 13.3 复用与归属原则

1. **核心代码零复用**：不复制 LangChain、LangGraph、AutoGen、Semantic Kernel、Temporal 的 loop、自动工具执行、状态机或审批实现；这些均只用于设计对照。
2. **允许的基础依赖**：仅使用 `openai`、`httpx`、`pydantic`、`pytest` 等不接管核心编排的库。
3. **小型辅助片段也需核验**：若后续确有必要借用某段非核心代码，先确认许可证、保留所需 notice，并在 README 的 Attribution 小节标明来源与修改范围。
4. **最终提交强调自主性**：README 会明确写出 “model proposes; harness validates, schedules, persists, and decides approval”，并链接到 `loop.py`、`scheduler.py`、`policy.py`、`state_store.py` 作为证据。

## 14. GitHub 调研的具体借鉴与限制

| 参考 | 借鉴内容 | 在本项目中的处理 |
|---|---|---|
| `aafre/agentharness` | 将副作用记录为 event、状态与记录可检查/回放 | 不安装该包；自写简化的 JSONL event 与 RunState。 |
| `nderman/agent-harness` | 轨迹断言、预录模型响应、故障注入测试 | 不复用 Agent runtime；采用 MockLLM 和 pytest fixture。 |

## 15. 6 小时实施优先级

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

## 16. 验收标准

提交前逐项确认：

- [ ] 不使用任何替代项目来执行 Agent loop、工具调度或状态机。
- [ ] LLM 至少在真实 demo 中完成一次“工具调用 → 结果回填 → 下一轮模型调用”。
- [ ] 同一模型响应中的独立 I/O 工具发生真实并发，且有耗时/trace 证据。
- [ ] 一个工具失败不会取消其余独立工具，错误会回到 LLM。
- [ ] `write_report` 在批准前绝不执行；重启后可以批准或拒绝并恢复。
- [ ] 每次运行保留状态与事件轨迹；pytest 覆盖核心状态转移。
- [ ] README 让第三方可配置、运行、复核 demo 与理解边界。

