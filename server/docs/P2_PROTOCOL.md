# P2 协议：`S2LW` 关键点窗口上传

P2 中手机保留 M1/M2/M4/M6，服务器执行 M3（预处理、分类、8 次预测多数投票）与 M5（组句）。本文定义 M2→M3 的跨端对象 `LandmarkWindow` 和 M3→M4 的 `RecognitionEvent`。M4→M5 继续使用 P1 的 `POST /translate`。

## 端点

`POST /recognize`，与 `/translate` 同一 FastAPI 进程、同一端口 8080。

请求头：

```
Authorization: Bearer <S2S_TOKEN>        # 与 /translate 相同
Content-Type: application/octet-stream
Accept: application/json
X-S2S-Session-Id: <8 位 appSessionId>     # 必填
X-S2S-Request-Id: m3-<n>                  # 可选，回声到响应和日志
X-S2S-Model-Version: <classifier version> # 可选；严格模式下不匹配返回 409
```

会话键只用 `X-S2S-Session-Id`。M5 的 `session_id` 是 `appSessionId-epoch`，两者不要混用。

## 二进制请求体

所有整数和 float32 均为小端。

### 40 字节头

| 偏移 | 大小 | 字段 | 说明 |
|---:|---:|---|---|
| 0 | 4 | magic | ASCII `S2LW` |
| 4 | 1 | version | `1` |
| 5 | 1 | header_bytes | `40` |
| 6 | 2 | point_count | `118` 或 `543` |
| 8 | 2 | coord_count | `3`（默认）或 `2` |
| 10 | 2 | frame_count | `1..64`；手机通常在 `10..64` |
| 12 | 2 | flags | bit0 最新帧有手；bit1 reset_vote；bit2 543 布局；其余为 0 |
| 14 | 2 | reserved | `0` |
| 16 | 8 | window_id | 进程内单调；clear 不重置；允许跳号 |
| 24 | 8 | episode_id | 单调；服务端见到更大值就重置投票历史 |
| 32 | 8 | first_frame_id | 窗口中最旧帧的 frame_id |

### 每帧记录

| 大小 | 字段 |
|---:|---|
| 8 | `frame_id` u64 |
| 8 | `timestamp_ms` u64 |
| `mask_bytes` | 有效性掩码 |
| `point_count * coord_count * 4` | float32，point-major |

`mask_bytes = align4(ceil(point_count / 8))`：118 → 16，543 → 68。
`frame_stride = 16 + mask_bytes + point_count * coord_count * 4`。

掩码按字节 LSB-first：字节 `j` 的 bit `b` 对应点 `j*8+b`，bit 1 = 存在。`point_count` 之后的填充位必须为 0。

缺失点的坐标必须写 `0.0`；服务端在预处理前恢复为 NaN。帧有效性 = 掩码任意位为 1，等价于 Android 的“任意 x 非 NaN”。

### 尺寸

| 布局 | 单帧 | 64 帧 body |
|---|---:|---:|
| 118 点 / 3 坐标 | 1,448 B | 92,712 B（90.54 KiB） |
| 118 点 / 2 坐标 | 976 B | 62,504 B（61.04 KiB） |
| 543 点 / 3 坐标 | 6,600 B | 422,440 B（412.54 KiB） |

开题报告 §4.2 的 543 点纯数据估算为 `64 × 543 × 3 × 4 = 417,024 B = 407.25 KiB`，不含头和掩码。P2 基线使用 118 点 / 3 坐标，约减少 4.5×。

## 响应（`RecognitionEvent`）

```json
{
  "schema": 1,
  "request_id": "m3-42",
  "session_id": "a1b2c3d4",
  "window_id": 42,
  "episode_id": 7,
  "status": "ok",
  "inferred": true,
  "label": "hello",
  "confidence": 0.87,
  "top3": [{"label": "hello", "confidence": 0.87}, {"label": "happy", "confidence": 0.06}],
  "has_hand": true,
  "frame_count": 64,
  "queue_ms": 1.2,
  "decode_ms": 0.4,
  "preprocess_ms": 2.1,
  "inference_ms": 18.3,
  "vote_ms": 0.05,
  "total_ms": 22.05,
  "model_version": "islr-conv1d-transformer-v1",
  "server_instance": "1a2b3c4d"
}
```

`status`：

| 值 | 含义 | 手机行为 |
|---|---|---|
| `ok` | 完成推理与投票 | 更新当前标签/置信度，进入 M4 |
| `insufficient_frames` | 有效帧 < 3，未推理 | **不得**改动当前标签（对应本地早退） |

`inferred=false` 时 `label=""`、`confidence=0`、`top3=[]`。`confidence` 是 top-1 softmax 概率，不是票数比例；与本地 `SignRecognizer` 一致。

## 服务端会话规则

每个 `session_id` 维护：`episode_id`、`last_window_id`、8 长预测历史、当前 label/conf/top3、`last_seen`。

单 worker 按 FIFO 处理：

1. `episode_id < state.episode_id` → 409 `stale episode`（旧会话迟到请求不能把状态拉回）。
2. `episode_id > state.episode_id` → 清历史与 `last_window_id`，更新 episode。
3. `window_id <= state.last_window_id` → 409 `stale window`；允许跳号（手机丢窗）。
4. 更新 `last_window_id`；`reset_vote` 只清历史，不改 episode。
5. 解锁后做预处理/推理；若有效帧 < 3，返回 `insufficient_frames` 且不改 label/history。

投票严格对齐 Kotlin：`topConf >= 0.2 && has_hand` 才入历史；多数票平局取先出现者；`!has_hand` 清历史；top-3 用稳定排序。

TTL 600 s、最多 32 个会话。超时 3 s 后 API 返回 504 并取消 future，但 worker 仍可能完成并更新投票状态；手机不重试同一 `window_id`。

## 状态码

| 码 | 场景 |
|---:|---|
| 200 | 处理完成（含 `inferred=false`） |
| 400 | header/body 长度、magic、version、first_frame_id 不匹配 |
| 401 | token 错误 |
| 409 | 旧 episode、重复 window、严格模式版本不匹配 |
| 413 | body 超过 `P2_MAX_BODY_BYTES` |
| 422 | point_count/coord_count 不支持、掩码填充位非零、未知 flag |
| 503 | 分类器未加载、队列满 |
| 504 | 分类超时 |

## 日志与对拍

`p2.jsonl` 每行记录：`session_id`、`request_id`、`window_id`、`episode_id`、`frame_count`、`point_count`、`coord_count`、`bytes`、`body_sha256`、`status`、`inferred`、`label`、`confidence`、`has_hand`、`vote_len`、`queue_ms`、`decode_ms`、`preprocess_ms`、`inference_ms`、`vote_ms`、`total_ms`、`http_status`、`error`、`model_version`、`server_instance`。

手机在 `m3_window_submitted` 里记录同一 `body_sha256`；两端哈希一致即可证明端序、stride、掩码顺序和点的选取一致。需要进一步定位时设置 `P2_DUMP_WINDOWS=1`，服务端会把原始 body 落到 `P2_DUMP_DIR`，可交给 PC harness 的 `preprocess` 复现。

## 计时口径

- `queue_ms`：入队到 worker 出队。
- `decode_ms`：解析 40 字节头与帧记录。
- `preprocess_ms`：118/543 点选取、归一化、dx/dx2、补齐。
- `inference_ms`：TFLite `invoke`。
- `vote_ms`：8 长历史更新与 top-3。
- `total_ms = queue + decode + preprocess + inference + vote`。
- 手机 `client_ms` 从发请求到解析响应；网络开销 = `client_ms - total_ms`。
