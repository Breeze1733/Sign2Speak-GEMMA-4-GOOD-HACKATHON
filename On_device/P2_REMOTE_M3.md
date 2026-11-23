# P2 远程分类客户端（Android）

P2 把 M3（预处理 + TFLite 分类 + 8 次预测多数投票）和 M5（组句）放到服务器；手机保留 M1 采集、M2 关键点、M4 词元与断句、M6 呈现播报。协议见 `server/docs/P2_PROTOCOL.md`。

## 运行模式

右上角三态：

| 按钮 | M3 | M5 |
| --- | --- | --- |
| `P0 LOCAL` | 手机 `SignRecognizer` | 手机 Gemma |
| `P1 G2T` | 手机 `SignRecognizer` | 服务器 `/translate`，失败回退本地 |
| `P2 M3` | 服务器 `/recognize` | 服务器 `/translate`，不回退 |

`CFG` 填同一个 base URL 和 token；P2 自动使用 `<base>/recognize`，P1 使用 `<base>/translate`。选择持久化在 SharedPreferences（`pipeline_mode`），旧 `mode` 键会迁移：`remote→p1`、`local→p0`。

P2 冷启动**不加载** 2.58 GB Gemma，也不构造本地 `SignRecognizer`；因此 P2 只需要服务器可达，不需要手机上的模型文件。

## 改动

| 文件 | 内容 |
| --- | --- |
| `LandmarkLayout.kt`（新） | 543/118/708/64 常量、`POINT_LANDMARKS` 与模型版本 |
| `HolisticAssembler.kt`（新） | 从 `SignRecognizer` 原样搬出的 M2 组装与 overlay |
| `FrameWindow.kt`（新） | 64 帧窗口与 `WindowSnapshot`（118 点 × 2/3 坐标 + 掩码） |
| `P2Protocol.kt`（新） | `S2LW` v1 小端编码，40 字节头 |
| `RemoteSignRecognizer.kt`（新） | 异步单飞、HTTP、epoch/window 过滤、无手处理 |
| `RecognitionController.kt`（新） | 统一 Local/Remote controller 与 `RecognitionEvent` |
| `SignRecognizer.kt` | 常量/组装/overlay 委托；新增非 timing 也递增的 `inferenceCount` |
| `MainViewModel.kt` | 三态模式、controller 切换、逐帧/异步拆分、`handleRecognition` |
| `TranslationConfigStore.kt` | `pipeline_mode` 与迁移 |
| `GlossToText.kt` | `RemoteConfig` 支持 base URL 或完整 `/translate`/`/recognize` |
| `ui/CameraScreen.kt` | P0/P1/P2 三态按钮、M3 状态行 |
| `MainActivity.kt` | P1/P2 自动跳过 Gemma 下载门禁 |
| `Gloss2TextEngine.kt` | `close()` 复位 `isInitialized`，支持 P2→P0/P1 切换后重新初始化 |

## 识别状态机

- **窗口**：`FrameWindow` 保留最近 64 帧；每帧 `HolisticAssembler.assemble` 生成 543×3（缺失为 NaN）。
- **尝试节奏**：`frameSeq % 3 == 0 && window.size >= 10`，与本地 `SignRecognizer` 一致。
- **单飞**：`inFlight.compareAndSet(false,true)` 成功才快照/编码/发送；忙时写 `m3_window_dropped`，`windowId` 仍然递增，服务端接受跳号。
- **无手**：只在 `hasHand=true` 时上传，避免相机空转时持续上传 NaN 窗口。持续无手 ≥200 ms 时递增 episode、复位服务端投票，并本地合成一次空 `RecognitionEvent`，让 M4 推送最后一个稳定标签。
- **epoch**：`clearBuffer`、3 秒自动 flush、模式切换、持续无手都会递增 `episodeId`；旧 epoch 的响应只记 `m3_window_stale`，不显示、不播报。
- **手动 TRANS**：`suppressBefore(lastWindowId)`，只丢弃提交前的在途结果，不重置服务端投票。
- **失败**：超时/HTTP 错误只记事件和状态，不做本地 M3 回退；P2 的 M5 也不回退本地 Gemma，失败时显示词元。

## 性能事件（logcat tag `S2S_PERF`）

- `m3_window_submitted`：`window_id`、`episode_id`、`frame_count`、`bytes`、`body_sha256`、`has_hand`。
- `m3_window_returned`：`status`、`label`、`confidence`、`client_ms`、`server_queue_ms`、`server_preprocess_ms`、`server_inference_ms`、`server_vote_ms`、`server_total_ms`、`bytes_down`。
- `m3_window_dropped`、`m3_window_timeout`、`m3_window_error`、`m3_window_stale`。
- `recognition_result` 增加 `source`、`window_id`、`classification_call_ms`；`word_recognized` 增加 `source`。

`body_sha256` 与服务器 `p2.jsonl` 中的同名字段一致，可直接用于端序、stride、掩码和点选取的对拍。

## 运行

1. 构建计时包：
   ```powershell
   .\gradlew.bat :app:assembleDebug -Psign2speakTiming=true -Ps2sRemoteUrl="http://121.41.19.165:8080"
   ```
2. 安装后进入相机页，点 **`P2 M3`**，`CFG` 填 base URL 与 token。
3. 选一次 `.BIN`，点 `REPLAY`。`run_trials.py` 仍可用，模式需先手动选好。

## 已知限制

- 118 点 / 3 坐标的完整窗口为 92,712 B；在 10 窗/s 的满速下约 7.4 Mbps，1/5 Mbps 条件会严重丢窗。协议保留 `coord_count=2` 和增量上传的扩展位，本次未启用。
- 单飞会丢弃忙时窗口（本次 26%），识别更新频率低于本地逐帧推理；这是 P2 通信/延迟权衡的一部分，不是错误。
- 服务端超时后 worker 仍可能完成并更新投票；手机不重试同一 `window_id`。
- 无手时不上传是相对提案“上传完整窗口”的刻意偏差，用于避免空转流量；服务端用 episode 变化清历史。
- 本次只做无整形单条件基线，未做完整 RTT/带宽矩阵。
