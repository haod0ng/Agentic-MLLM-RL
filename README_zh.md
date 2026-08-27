# 基于 Relax 的 Agentic MLLM RL

本仓库是 [Relax](https://github.com/redai-infra/Relax) 的研究快照，增加了面向多模态 Agentic RL 的两个本地
Reward-Model/Judge 服务：结果 Judge（`answer_accuracy`）与多模态过程/轨迹 Judge
（`multi_turn_reasoning`）。两个服务作为独立 Ray Serve Deployment，可部署到专用 GPU 组。

过程 Judge 支持两种执行方式：

- `terminal_once`：episode 结束后读取完整轨迹并评分一次；
- `per_turn`：每次完成 `response -> observation` 交互后异步评分，最终取均值形成一个 trajectory-level
  reward 分量。

当前 `per_turn` 不产生 turn/token 级 advantage；最后一个 assistant response 没有后续 observation，因此不被
per-turn 过程 Judge 覆盖，只由结果 Judge 覆盖。

## 主要内容

- `relax/engine/rewards/`、`relax/agentic/session/`：双 Judge reward、projection 与上下文；
- `relax/components/`、`relax/distributed/ray/`：Ray Serve 部署、placement 与并发控制；
- `examples/agentic_dual_judge/`：通用配置与 latency analyzer；
- `examples/mobilegym_agentic/`：MobileGym adapter、启动模板与评测工具；
- `tests/agentic/`、`tests/engine/rewards/`：CPU/unit test 与 opt-in GPU test。

完整 Relax 源码被保留，以便在真实集成上下文中审查和验证。上游来源及公开边界见
[PROVENANCE.md](PROVENANCE.md)。

## 关键语义边界

`benchmark_mode=dual` 时训练 reward 为：

```text
score = 0.8 * answer_accuracy + 0.2 * multi_turn_reasoning
```

MobileGym adapter 返回 `reward=None`，使两个本地 Judge 成为训练 reward 来源。结果 Judge 可以读取
`field/expected/actual` 形式的终态检查证据，因此应称为“state-check-assisted outcome Judge”，不能简单称为
final-answer-only ORM。

如果要做纯系统 latency 对照，应使用 `recorded` 或 `dual_shadow`，冻结训练 reward 和 workload。在线比较
`dual + terminal_once` 与 `dual + per_turn` 会同时改变 reward 语义和后续 policy trajectory，只能解释为联合系统
效应。

## 快速验证

```bash
python -m pip install -r requirements.txt
python -m pip install \
  "transferqueue @ git+https://github.com/redai-infra/TransferQueue.git@58054a33834aadbcf76aacd6b1e32e25c030f2c9" \
  --no-deps
python -m pip install -e .
pytest -q \
  tests/engine/rewards/test_dual_local_judge.py \
  tests/engine/rewards/test_reward_projection.py \
  tests/utils/test_judge_config.py \
  tests/agentic/session/test_reward_context.py \
  tests/agentic/pipeline/test_reward_dual_judge.py
```

TransferQueue 是当前 Controller 的必需公开 Git 依赖，但上游 `requirements.txt` 尚未声明；这里的 revision 与
`relax/core/controller.py` 的兼容性检查一致。完整 Ray/SGLang/Megatron 训练仍应使用项目容器。

GPU 集成测试需要目标 Ray/SGLang 环境和本地模型，不属于默认 CI：

```bash
pytest -q -m gpu tests/integration/test_agentic_dual_judge_real_models.py
```

历史 latency 数据只作为系统诊断案例，不能证明 reward 是唯一因果瓶颈，也不能证明 GPU idle 的来源或
`max_concurrency` 的收益。修正后的证据分级见
[EVALUATION_REPORT.md](examples/mobilegym_agentic/EVALUATION_REPORT.md)。原始轨迹、截图、checkpoint 与集群日志
不在本仓库中。

本仓库基于 Relax commit `d52cd0aca9b347a57fb435bda3ae2db8fc6706a4`，采用 Apache License 2.0。详见
[NOTICE](NOTICE) 与 [PROVENANCE.md](PROVENANCE.md)。
