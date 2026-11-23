# P2 远程分类 · 服务器侧基线（2026-11-24）

## 环境

| 项 | 值 |
| --- | --- |
| 机器 | 阿里云 ECS，Ubuntu 22.04.5，4 vCPU / 7.3 GiB / 40 GB，无 GPU |
| 服务器 IP | 121.41.19.165:8080，systemd `sign2speak-p1`（同一进程同时提供 `/translate` 与 `/recognize`） |
| 分类器运行时 | `ai-edge-litert==2.4.0`，CPU，`CLASSIFIER_THREADS=1`；`numpy==2.2.6` |
| 分类器模型 | `classifier.tflite`，7,057,180 字节，sha256 `08d0af17c7dcf3a690b9228415513ded5452d3fcd1889ef9268db659e3fc1bcb` |
| 模型元数据 | `model_meta.json`，`num_classes=50`，输入 `(1,64,708)`，输出 `(1,50)`，`model_version=islr-conv1d-transformer-v1` |
| 预热 | 首个零输入推理 5.03 ms |
| 组句模型 | `gemma-4-E2B-it.litertlm`，2,588,147,712 字节，LiteRT-LM 0.19.0 CPU 4 线程 |

## 分类器接入门禁

`scripts/check_classifier.py` 通过：

- 运行时 `ai-edge-litert`，输入 `(1,64,708)`，输出 `(1,50)`；
- 合成窗口 top-3：`see 0.142`、`look 0.101`、`night 0.093`（随机窗口，仅用于确认推理通路）；
- 有效帧 < 3 时返回 `insufficient_frames`，不推理。

`SELECT_TF_OPS` 在 Linux 上可用，无需回退到 `tensorflow-cpu`。

## 单窗延迟（空载 vs Gemma 生成中）

本地回环 HTTP，10 个合成 64 帧窗口，会话内 window_id 递增；并发场景同时跑一条约 4.7 s 的 `/translate`。

| 场景 | client P50 / P95 (ms) | server total P50 / P95 (ms) | inference P50 / P95 (ms) |
| --- | ---: | ---: | ---: |
| 空载 | 9.32 / 13.40 | 6.71 / 7.36 | 3.93 / 4.16 |
| 与 Gemma 生成并发 | 14.58 / 16.00 | 10.66 / 12.27 | 6.23 / 7.29 |

并发 `/translate` 返回 200、`status=generated`、`inference_ms=4722.35`。Gemma 占满 4 线程时，单窗分类推理约 1.6× 空载，仍远低于 3× 的告警阈值；不需要降低 Gemma 线程数。

## `/recognize` 冒烟

- 合法窗口：200，`decode_ms=1.30`、`preprocess_ms=2.33`、`inference_ms=4.07`、`vote_ms=0.10`、`total_ms=7.93`。
- 重复 `window_id`：409 `stale window`。
- 截断 body：400，返回期望长度。
- `body_bytes=92,712`（118 点 / 3 坐标 / 64 帧）。

## 口径与限制

- 单窗数字是服务器本地回环，不含手机→服务器的网络 RTT；手机侧基线见 `p2-phone-baseline.md`。
- 合成窗口的标签没有语义意义，只用于验证解码、预处理、推理、投票和计时链路。
- 服务器是 CPU 后端，手机 P0 是 GPU 后端；按开题报告 §9.1 第 4 条，这属于“不同后端配置”，不能当作公平的硬件对照。
- 分类器与 Gemma 共享 4 vCPU；若后续并发场景明显恶化，可把 `CLASSIFIER_THREADS` 或 Gemma `THREADS` 作为独立配置记录。
