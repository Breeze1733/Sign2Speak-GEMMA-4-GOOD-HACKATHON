# P1 远程组句 · 服务器侧基线（2026-11-23）

## 环境

| 项 | 值 |
| --- | --- |
| 机器 | 阿里云 ECS，Ubuntu 22.04.5，Intel Xeon Platinum，4 vCPU / 7.3 GiB / 40 GB |
| 服务器 IP | 121.41.19.165，服务端口 8080 |
| 推理栈 | `litert-lm-api==0.19.0`（与 Android `litertlm-android:0.19.0` 对齐） |
| 模型 | `gemma-4-E2B-it.litertlm`，2,588,147,712 字节 |
| 模型 sha256 | `181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c` |
| 后端 | CPU，4 线程 |
| 采样参数 | top_k=40, top_p=0.95, temperature=0.3（照搬 Gloss2TextEngine.kt） |
| 提示词 | 与 `Gloss2TextEngine.kt` 逐字一致 |
| 模型加载耗时 | 冷启动 5.8 s；服务重启且模型已在页缓存时 0.5 s |

## 结果

单实例、单 worker 串行推理，10 条不同词元，全部 `status=generated`：

| 指标 (inference_ms) | 值 |
| --- | ---: |
| 平均 | 4033.9 |
| 中位 | 3996.5 |
| 最小 | 3748.3 |
| 最大 | 4539.1 |
| P95 | 4173.6 |

3 条并发请求验证串行队列（`queue_ms` → `inference_ms`）：

| request | queue_ms | inference_ms | total_ms |
| --- | ---: | ---: | ---: |
| q1 | 0.12 | 4145.8 | 4145.9 |
| q3 | 4143.2 | 4162.5 | 8305.7 |
| q2 | 8305.7 | 4160.4 | 12466.0 |

生成样句（与手机端示例一致）：

- `WATER PLEASE` → "Can I have some water, please?"
- `HUNGRY FOOD PLEASE` → "I'm hungry. Could I have some food, please?"
- `WHERE MOM` → "Where is my mom?"
- `GO SLEEP NOW` → "I'm going to sleep now."
- `SICK` → "I feel sick."

## 对照与口径

- 手机端 P0（2026-11-02，小米 8 / SD845，Gemma GPU）文本生成：单词平均 5.57 s，多词平均 5.53 s。
- 服务器 CPU 本组样本平均 4.03 s，**不构成公平对照**：手机端是 GPU 后端、服务器是 CPU 后端，且样本、机型、输入不同。按开题报告 9.1 第 4 条，这属于"不同后端配置"，需单独记录。
- 本表只含服务器侧 `inference_ms`，不含手机→服务器的网络 RTT、约 3 秒自动断句等待、界面呈现与语音播放。

## 复现

```bash
TOKEN=$(grep '^S2S_TOKEN=' /opt/sign2speak/app/.env | cut -d= -f2)
curl -s -X POST http://127.0.0.1:8080/translate \
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d '{"request_id":"r1","session_id":"s","sentence_index":0,"glosses":["WATER","PLEASE"],"style":"friendly"}'
```

原始记录：`/opt/sign2speak/results/p1-runs.jsonl`；请求日志：`/opt/sign2speak/logs/p1.jsonl`。
