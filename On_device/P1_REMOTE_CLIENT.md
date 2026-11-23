# P1 远程组句客户端（Android）

给 App 加了 Local / Remote 两套组句实现，运行时切换，对应开题报告 7.1、7.2。

## 改了什么

| 文件 | 内容 |
| --- | --- |
| `GlossToText.kt`（新） | `GlossToText` 统一接口；`LocalGlossToText`（包 LiteRT-LM）；`RemoteGlossToText`（HTTP `/translate`） |
| `TranslationConfigStore.kt`（新） | SharedPreferences 保存模式、URL、token |
| `MainViewModel.kt` | 按配置选择实现；远端失败回退本地；会话版本 + 句子序号；串行化保证顺序；计时埋点 |
| `ui/CameraScreen.kt` | 右上角 `G2T LOCAL / G2T REMOTE` 切换、`CFG` 配置弹窗、上次来源显示 |
| `AndroidManifest.xml` | `usesCleartextTraffic="true"`（服务是 http） |
| `build.gradle.kts` | `BuildConfig.REMOTE_URL / REMOTE_TOKEN / REMOTE_MODEL_VERSION` |

## 用法

0. **没有本地模型也能进**：下载页底部有 **“Skip — recognition / remote P1 only”**，点了直接进相机页。远程 P1 不需要 2.58 GB 模型。
1. 进相机页，右上角有 **`G2T LOCAL`**（棕色）或 **`G2T REMOTE`**（青色），点一下切换。
2. 点 **`CFG`** 填服务地址和 token：
   - Base URL：`http://121.41.19.165:8080`
   - Token：与服务端 `.env` 里的 `S2S_TOKEN` 一致
   - Token 留空表示保持原值
3. 下方小字显示上一次结果来源与状态，例如 `last: remote generated`。

默认 Base URL 来自 Gradle 属性，构建时可覆盖：

```powershell
.\gradlew.bat :app:assembleDebug -Ps2sRemoteUrl="http://121.41.19.165:8080" -Ps2sRemoteToken="<token>"
```

不传 `-Ps2sRemoteToken` 时 token 为空，需要在 App 的 `CFG` 里填。

## 行为（对应提案）

- **统一接口**：`MainViewModel` 只调 `GlossToText.translate(...)`，不关心本地还是远端。
- **请求内容**：`request_id`、`session_id`、`sentence_index`、`glosses`、`style`、`model_version`。
- **顺序**：用 `Mutex` 串行化，句子按提交顺序返回和播报。
- **会话版本**：清空/重放时 `sessionEpoch++`，旧会话返回的响应只记日志（`stale_response`），不显示不播报。
- **超时与回退**：连接 5 s、读取 60 s；远端失败自动回退本地模型；本地模型也不可用时显示词元并标记失败。
- **计时**：服务端 `queue_ms / inference_ms` 和客户端 `client_ms` 都写进 `S2S_PERF` 日志。

## 日志事件（logcat tag `S2S_PERF`）

`translation_mode`、`remote_request`、`remote_response`、`remote_http_error`、`remote_timeout`、`remote_error`、`remote_fallback`、`stale_response`、`sentence_returned`（带 `source` / `status` / `client_ms` / `server_queue_ms` / `server_inference_ms`）。

采集仍用 `performance-tools/collect.mjs`。

## 注意

- 服务是 **http 明文**，token 在网络上明文传输；实验结束记得关端口或换 TLS。
- 远端不带本地模型也能用（组句在云端）；只有回退本地时才需要手机上的 2.58 GB 模型。
- 远端 60 s 超时会占用串行队列，实验中如果网络差，建议先按提案的 10/50/100 ms 条件测，不要长时间挂死。
