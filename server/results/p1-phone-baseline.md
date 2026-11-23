# P1 手机端基线（远程组句）· 2026-11-24

服务器侧基线见 [p1-server-baseline.md](p1-server-baseline.md)。本文记录**手机端发起 P1 请求**的实测数据，供与 P0 本地基线对照。

## 环境

| 项 | 值 |
| --- | --- |
| 手机 | OPPO Find X9 Ultra（PMA110），Android 16 / API 36，arm64-v8a |
| 应用 | Sign2Speak Timing 1.0，包名 `com.sign2speak.app.perf`，`TIMING_ENABLED=true` |
| 输入 | 本地帧包回放 `asl_test_hello_hungry_water_thankyou.bin`（640×480，30 fps，488 帧，16.27 s） |
| 网络 | 5G 移动数据 |
| 服务 | `http://121.41.19.165:8080/translate`，LiteRT-LM 0.19.0，CPU 后端 |
| 组句模式 | `G2T REMOTE`（M1–M4 在手机，M5 在云端） |
| 测试时间 | 2026-11-24 00:50–00:54 |

帧包是 4 个不同手语者的拼接片段，约每 4 秒硬切一次。

## 结果 1：手机端识别（M1–M4）

计时口径：`word_recognized` 事件 `t_ns` − `word_started.hand_seen_ns`，即"首次检测到手 → 置信度达到 50%"。

| 词序 | 词元 | episode | 识别耗时 (ms) | 置信度 | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | hello | 5 | 742.55 | 0.695 | 首词 |
| 2 | hungry | 6 | 795.52 | 0.601 | 后续词 |
| 3 | please | 7 | 8.35 | 0.514 | **< 100 ms，按测试口径属"过早输出"** |
| 4 | water | 8 | 71.42 | 0.793 | **< 100 ms，按测试口径属"过早输出"** |

按既有的"后续词过早输出"规则（首个达标输出距进手 <100 ms 时剔除），有效识别样本为 **首词 742.55 ms、后续词 795.52 ms**。

## 结果 2：远程组句（M5）

计时口径：`sentence_submitted` → `sentence_returned`，不含约 3 秒自动断句等待、界面呈现和语音播放。

| 轮次 | request_id | 词元 | 服务端排队 (ms) | 服务端推理 (ms) | 客户端总耗时 (ms) | 提交到返回 (ms) | 网络+客户端开销 (ms) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| trial 2 | 1 | hello hungry please water | 0.12 | 4759.29 | 4870 | 4874.55 | ≈115.1 |
| trial 3 | 2 | hello hungry please water | 0.10 | 4589.72 | 4710 | 4712.04 | ≈122.2 |

生成句子（两轮一致）：`Hello, I'm hungry. Can I have some water, please?`

服务器 `p1.jsonl` 对应记录：

```json
{"request_id":"1","session_id":"7f683fc7-3","glosses":["hello","hungry","please","water"],
 "status":"generated","queue_ms":0.12,"inference_ms":4759.29,"total_ms":4759.41}
{"request_id":"2","session_id":"7f683fc7-6","glosses":["hello","hungry","please","water"],
 "status":"generated","queue_ms":0.1,"inference_ms":4589.72,"total_ms":4589.82}
```

## 与 P0（全本地）对照

P0 数据来自《Sign2Speak 本地性能测试报告》（2026-11-02，小米 8 / MI 8 UD，Android 10，SD845，Gemma GPU，单词 20 次 + 多词 20 轮）。

| 指标 | P0 全本地（MI 8） | P1 远程组句（Find X9 Ultra） | 差额 |
| --- | ---: | ---: | ---: |
| 文本生成·单词 | 平均 5.57 s（5.34–6.28） | — | — |
| 文本生成·多词 | 平均 5.53 s（4.21–6.45） | **4.71–4.87 s**（2 次） | 约 −0.7 ~ −0.8 s |
| 词元识别·首词（多词） | 平均 218.44 ms | 742.55 ms（1 次） | +524 ms |
| 词元识别·后续词 | 平均 2036.09 ms | 795.52 ms（1 次有效） | −1241 ms |

**这些数字不能当作同条件对照：**

- **设备不同**：P0 是小米 8，P1 是 OPPO Find X9 Ultra；识别耗时差异主要来自设备，不来自划分方式。
- **后端不同**：P0 手机 GPU，P1 服务器 CPU，属"不同后端配置"，按开题报告 9.1 第 4 条单独记录。
- **输入不同**：P0 是现场手语；P1 是 4 人拼接帧包，硬切会制造 <100 ms 的过早输出。
- **样本量**：P1 只有 2 轮组句、1 组有效词元，不构成统计结论。

可以在报告中使用的**同口径结论**只有一条：在这台服务器（CPU）和这条 5G 链路上，"词元提交到文本返回"为 4.71–4.87 s，其中服务端推理占 4.59–4.76 s，网络+客户端约 0.12 s；该值低于 P0 在 MI 8 上的 5.53–5.57 s，但受设备与后端差异影响，需在受控实验中复核。

## 复现

```powershell
# 1) 计时包已安装：com.sign2speak.app.perf
# 2) 帧包位置
adb push testdata/packs/asl_test_hello_hungry_water_than...
  /sdcard/Android/data/com.sign2speak.app.perf/files/
# 3) App 内：Skip → G2T REMOTE → CFG 填 URL/token → 清空并开始新一轮测量 → CLIPS 选片段
# 4) 采集
adb logcat -s S2S_PERF
```

关键事件：`sentence_submitted`、`generation_started`、`remote_request`、`remote_response`、`sentence_returned`、`word_started`、`word_recognized`、`word_hand_removed`。

## 数据留存

- 手机侧事件：logcat tag `S2S_PERF`（trial 2、trial 3）
- 服务端请求：`/opt/sign2speak/logs/p1.jsonl`
- 帧包与清单：[testdata/manifest.csv](../../testdata/manifest.csv)
