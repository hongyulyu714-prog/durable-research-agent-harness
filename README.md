# Durable Research Agent Harness

一个从零实现的、支持工具调用、真实异步并发、人工审批与可恢复状态的 Python 研究 Agent 原型。

本项目的 Agent loop、工具调度、状态机和 Human-in-the-loop 由项目自身代码控制；`openai` 仅作为基础 OpenAI-compatible API 客户端使用。

> 当前仓库已完成工程骨架与设计文档，核心实现将在后续提交中逐步加入。

## 快速开始（开发中）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
research-agent --help
```

## 仓库结构

```text
src/research_agent/      # Harness 核心：loop、状态、策略、调度器和工具
tests/                   # 离线、确定性的单元与集成测试
data/                    # 只读演示资料
reports/                 # Agent 批准后生成的报告（默认不提交）
runs/                    # checkpoint 和 JSONL 运行轨迹（默认不提交）
examples/                # 可复现实验和 demo 输入
docs/                    # 原题、设计方案、后续运行记录
```

## 设计文档

- [详细实现方案](docs/Agent_Harness_详细方案.md)
- [原始题目](docs/Take-home试题（智能体方向）.md)

## 计划中的 CLI

```bash
research-agent run "阅读资料并生成研究简报"
research-agent resume <run_id> --approve
research-agent inspect <run_id>
pytest -q
```

## 依赖与边界

| 依赖 | 用途 | 是否控制核心执行链 |
|---|---|---:|
| `openai` | 基础 LLM API 调用 | 否 |
| `httpx` | 异步网页请求 | 否 |
| `pydantic` | 数据和参数校验 | 否 |
| `pytest` | 测试 | 否 |

不使用 LangChain、LangGraph、OpenAI Agents SDK 或其他 Agent/Harness 框架。

## 网页工具边界

`fetch_url` 只读取 `http` / `https` 资源。请求发出前和每次重定向前都会拒绝
localhost、回环、链路本地、私网、未指定地址及常见 metadata-service 目标；域名能够
解析到这类地址时同样拒绝。响应通过流式读取，默认最多保留 1,000,000 字节并返回最多
20,000 个字符，避免单个工具调用把超大响应完整载入内存。被安全策略拒绝的调用会作为
不可重试的结构化 `unsafe_url` 工具失败回填给 Agent。

## 审批恢复与持久化边界

写入工具不会被普通执行器直接调用：Agent loop 在验证完写入调用后，先把原始
`tool_call_id`、工具名和参数原子保存到 `runs/<run_id>/checkpoint.json`，并暂停为
`waiting_approval`。`research-agent resume <run_id> --approve` 只会执行该持久化调用；
`--reject <reason>` 则把结构化 `approval_rejected` 结果回填给模型。

批准前会额外落盘 `approval_execution_started`。若进程在副作用发生后、结果 checkpoint
写入前中断，下一次恢复会拒绝自动重试该调用，要求人工检查目标再处置。这避免了把“至少一次”
误称为 exactly-once；本地文件系统仍无法在不依赖下游幂等键或事务存储的情况下提供真正的
端到端 exactly-once 保证。
