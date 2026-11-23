# 评测工具链

把"手机上跑一遍"变成"可重复、可汇总、能写进报告"的数据。

## 组成

| 文件 | 作用 |
|---|---|
| `run_trials.py` | 自动跑 N 次回放：点"清空并开始新一轮测量"→点 REPLAY→等待，同时开 collect.mjs |
| `stats.py` | 把一次或多次运行的 `events.ndjson` 汇总成 `report.md` + `summary.csv` |
| `join_server.py` | 把 App 日志和服务器 `p1.jsonl` 按 `request_id` 合并 |
| `netem.md` | 网络条件控制与校准 |

计时本身（`PerfTrace`）和原始采集（`performance-tools/collect.mjs`）是现成的，这里补的是编排、合并、统计、出表。

## 前置条件

1. **计时包**：`.\gradlew.bat :app:assembleDebug -Psign2speakTiming=true`
   包名 `com.sign2speak.app.perf`。不带这个参数就没有 `S2S_PERF` 日志。
2. 手机已装计时包，`adb devices` 只有一个 `device`。
3. App 里完成一次性配置：
   - 下载页点 **Skip**（远端 P1/P2 不需要 2.58 GB 模型）
   - 右上角选 **`P0 LOCAL`**、**`P1 G2T`** 或 **`P2 M3`**
   - **`CFG`** 填 `http://121.41.19.165:8080` 和 token
   - P2 还需要服务端 `classifier_loaded=true`（`curl /healthz`）
   - 点 **`.BIN`** 选一次素材（脚本无法代劳系统文件选择器）
4. 屏幕常亮、App 前台、后台清理干净；**充电状态全程一致**。

## 跑一个条件

```powershell
uv run python eval/run_trials.py `
  --condition p2-rtt0 --reps 30 `
  --clip-seconds 17 --settle-seconds 8 `
  --mode p2 --url http://121.41.19.165:8080
```

脚本会：
- 在 `eval/runs/<condition>_<时间戳>/` 下开一个采集目录，写 `meta.json`
- 每轮点"清空并开始新一轮测量"、点 REPLAY、等 `clip + settle` 秒
- 结束时提示分析命令

如果找不到按钮文字（Compose 语义没暴露），用 `--no-tap`，脚本会等你按回车。

## 分析

```powershell
uv run python eval/stats.py eval/runs
```

输出 `eval/runs/analysis/report.md`（按条件分组的中位数/P95 表）、`summary.csv`、`m3.csv`（逐窗口）和 `m3_summary.csv`（每轮窗口/丢窗/字节汇总）。

两端合并：

```powershell
uv run python eval/join_server.py `
  --events eval/runs/<run>/events.ndjson `
  --server <从服务器拷下来的 p1.jsonl> `
  --out eval/runs/<run>/joined.csv
```

## 产出的指标

| 指标 | 来源 |
|---|---|
| 词元识别 `hand_detected→50%` | `word_started` → `word_recognized` |
| 组句 `提交→文本返回` | `sentence_submitted` → `sentence_returned` |
| 客户端总耗时 | `sentence_returned.client_ms` |
| 服务器排队 / 推理 | `server_queue_ms` / `server_inference_ms`（或 p1.jsonl） |
| 网络开销 | `client_ms - queue - inference`（含 HTTP + 末段 RTT） |
| 处理帧率 | `recognition_result.frame_observed_ns` 间隔 |
| 成功率 / 超时 / 回退 / 过期响应 | `generation_status`、`remote_*`、`stale_response` |
| P2 窗口 `提交→返回` | `m3_window_submitted` → `m3_window_returned` |
| P2 服务端分段 | `server_queue_ms` / `preprocess_ms` / `inference_ms` / `vote_ms` / `total_ms` |
| P2 网络开销 | `client_ms - server_total_ms` |
| P2 丢窗 / 字节 | `m3_window_dropped`、`m3_window_submitted.bytes`、`m3_window_returned.bytes_down` |

## 已知缺口（代码侧）

1. **`RemoteGlossToText` 不发 `generation_status`**，导致远程句子在旧版 `sentences.csv` 里状态缺失。
   `stats.py` 会回退读取 `sentence_returned.status`，但 `collect.mjs` 的 CSV 仍是空。建议在
   `GlossToText.kt` 的远端分支补发一行 `generation_status`。
2. **两端 session 不同**：App 发的是 `appSessionId-epoch`，PerfTrace 日志的 `session` 是另一个 UUID。
   因此 `join_server.py` 按 `request_id` 合并；同一次采集内唯一，重启 App 会从 1 重新计数。
3. `run_trials.py` 只重复同一段素材；多素材随机顺序需要扩展参数（按 `--clips` 列表）。
4. **P2 的 M3 请求没有加入 `join_server.py`**：手机日志已带服务端分段耗时，单基线够用；需要与
   `p2.jsonl` 逐窗对拍时按 `window_id` + `body_sha256` 合并（服务端日志字段见
   `server/docs/P2_PROTOCOL.md`）。
5. **P2 无手窗口不上传**：这是为降低空转流量做的刻意偏差；服务端用 `episode_id` 变化清投票历史。
6. `run_trials.py` 仍不能自动切换 P0/P1/P2；模式需在 App 里先选好。
