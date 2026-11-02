# OpenChicken RL

MicroDuck / OpenChicken 机器人的仿真训练工程，包含原生 mjlab 任务，以及针对 V12 侧身靠动作的 Supertracking 世界模型和策略训练。

基于 [Pollen Robotics microduck_rl](https://github.com/pollen-robotics/microduck_rl) 的固定版本 `bae39f5900b318f7e87fc485c30a31213cfa14eb`。上游说明保存在 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)，来源与本地改动见 [UPSTREAM_SOURCE.json](UPSTREAM_SOURCE.json)。

## 环境

已验证：Linux x86_64、Python 3.12.13、PyTorch 2.9.1 CUDA 12.8、mjlab 1.3.0、MuJoCo 3.10、Warp 1.12、BAM 1.0.1。依赖版本和 BAM Git revision 以 `uv.lock` 为准。

```bash
cd openchicken_rl
source env.sh
uv sync --frozen
source env.sh
```

`env.sh` 支持 Bash / Zsh，激活本仓库 `.venv`，并把缓存放在仓库旁的 `.cache/external`。可通过 `OPENCHICKEN_RL_CACHE_ROOT` 指定其他用户可写缓存位置。虚拟环境和缓存不提交到 Git；从现有 OpenChicken 工作区迁移时，本机已安装环境会保留。

## Supertracking 训练

所有训练默认从头开始，模型、优化器、归一化统计和 Replay 都重新初始化。每次使用新的输出目录；只有明确需要恢复时才使用 `--resume`。

先运行 64 环境、5 轮短测：

```bash
python scripts/train_supertracking.py --robust --smoke \
  --batch-size 128 --wm-batches 4 --replay-device cuda:0 \
  --output logs/supertracking_smoke
```

当前 RTX 4070 Ti 12 GB 已验证 1024 环境、GPU Replay，保留原接触/约束缓冲配置。以下命令从头训练 20000 轮：

```bash
python scripts/train_supertracking.py --robust \
  --num-envs 1024 --iterations 20000 \
  --batch-size 128 --wm-batches 4 --replay-device cuda:0 \
  --output logs/supertracking_v12_1024_20k
```

默认学习参数：学习率 `2e-4`，每轮采集 8 步，Replay 容量 256 步/环境，世界模型拟合窗口 8 步，策略想象窗口 32 步，预热采集 40 步，前 30 轮只训练世界模型，每 100 轮评估和保存。上面的正式长训命令仅为运行说明；迁移仓库不会自动启动训练。

`--robust` 启用机械 Backlash、质量/惯量/质心/摩擦/电压等随机化、编码器固定偏差、IMU 安装偏差、观测噪声与延迟。训练关闭根部速度突变推扰；独立推扰测试可使用 `make_microduck_supertracking_env_cfg(robust=True, push_test=True)`。

BAM 命令延迟按每个 5 ms 物理步独立抽样 3～6 步处理。Replay 保存 7 格命令队列、有效长度、四个子步的实际 lag 和电机目标。世界模型根据实际收到的目标预测运动；想象 rollout 使用同样的可微队列。Critic 可见待执行命令，策略不读取未来随机 lag。

`--replay-device cpu`、`--constraint-capacity`、`--contact-capacity` 是可选显存调节项，默认不改变 GPU Replay 和原缓冲容量。缓冲溢出会报错停止。

## 参考动作与输出接口

- `motions/v12/motion.npz`：完成碰撞对齐的原始 10 秒 V12。
- `motions/v12_stand_transitions/motion.npz`：1 秒站立过渡 + 完整 V12 + 1 秒返回站立，共 12 秒；这是 `--robust` 的默认参考。
- 两份动作的 SHA-256、帧率和来源在同目录的 `manifest.json` 中。仓库移动不改变动作数据。

输出目录包含 `metrics.jsonl`、配置/来源记录、`best.pt`、`last.pt`、`policy.onnx` 和 `motion_latents_50hz.npz`。`best.pt` 按评估存活率及回报选择；默认导出的 ONNX 来自训练结束时的策略。日志和大检查点不提交到 Git。

Supertracking 策略接口为 `history[10,48] + motion_latent[32] -> action[14]`，动作是相对 HOME 的弧度目标；归一化包含在 ONNX 中。它与原生任务的 61D 观测接口不同，不能直接当作原生策略部署。本仓库的训练命令不连接或驱动实机。

显式 BAM 延迟版本使用 `bam_substeps_v1` 检查点契约，旧世界模型/Replay 不可直接续训。旧策略的推理接口没有变化。

## 目录

| 路径 | 用途 |
| --- | --- |
| `src/mjlab_microduck/supertracking/` | 环境封装、命令延迟、世界模型、策略、Critic、Replay 和训练器 |
| `src/mjlab_microduck/tasks/` | 原生任务、Supertracking 配置、奖励与随机化 |
| `src/mjlab_microduck/robot/` | 机器人模型和网格 |
| `src/mjlab_microduck/actuator/` | BAM 摩擦随机化与 Backlash 编码器模型 |
| `scripts/` | 训练、动作准备、物理检查、导出等工具 |
| `motions/` | 随仓库保存的两份参考动作 |
| `tests/` | 配置、运动学、延迟队列和训练组件测试 |
| `docs/` | 上游技术文档和说明 |

工作区外层的 `third_party/_staging` 是原始下载审核区；运行本仓库不依赖其中的源码副本。

## 验证

```bash
source env.sh
uv run --with pytest pytest tests/ -q
```

GPU 专项测试会构建离线仿真；无可用 CUDA 时跳过对应测试。GPU 测试覆盖延迟序列、局部 reset 隔离、编码器偏差、推扰后状态时序，以及 CPU/GPU Replay 抽样一致性。

## 许可证

保留上游训练代码的 [Apache-2.0 许可证](LICENSE)。Pollen Robotics 的 STL 模型遵循 [CC-BY-NC-SA-4.0](LICENSE-MODELS.md)。远端仓库初始化时的 MIT 原文保存在 [LICENSE-PROJECT-MIT](LICENSE-PROJECT-MIT)，不替代上游代码与模型各自的许可。模型来源和修改归属保持可追溯。
