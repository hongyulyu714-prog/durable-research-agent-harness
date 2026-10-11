# 可复现实验

主验收样例是 `tests/test_integration.py`，不依赖网络或模型 API：

```text
MockLLM
  -> read_file + fetch_url（并发；fetch_url 故障）
  -> MockLLM 生成 write_report
  -> durable checkpoint / waiting_approval
  -> 新 ApplicationService resume --approve
  -> write_report 一次
  -> MockLLM 最终回答 / JSONL trace / summary
```

执行：

```bash
pytest -q tests/test_integration.py
```

测试会在临时目录中创建 `runs/offline_integration/`，并断言脱敏轨迹含 `llm_response`、`tool_result`、`tool_batch_completed`、`approval_requested`、`approval_granted`、`tool_started`、`tool_finished` 和 `run_completed`。为避免将演示内容或本机路径纳入版本库，临时 checkpoint、trace 和报告不提交。

若已配置 `OPENAI_API_KEY` 与 `OPENAI_MODEL`，按仓库根目录 README 的 `run` / `resume` 命令进行真实模型演示。仅在审阅后从 `runs/<run_id>/` 复制 `events.jsonl` 与 `summary.json`；不要复制 API Key、原始网页内容或含私密任务的 checkpoint。
