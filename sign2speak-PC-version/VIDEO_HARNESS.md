# 视频识别 harness（PC，无摄像头）

`recognize_video.py` 把一段视频按手机 App 的同一套规则跑一遍：MediaPipe 关键点 → 118 点预处理 → `classifier.tflite` → 投票 → 词元接纳 → 3 秒无手断句，然后把词元发给云端 P1 组句服务。

## 与手机对齐的点

| 环节 | 取值 | 来源 |
|---|---|---|
| 面部检测频率 | 每 3 帧 | `LandmarkDetector.kt` |
| 手部检测 | 每帧，最多 2 只手，全分辨率 | 同上 |
| 滑动窗口 | 最长 64 帧，每 3 帧推理，至少 10 帧 | `SignRecognizer.kt` |
| 投票 | 近 8 次多数表决 | 同上 |
| 推理阈值 | top-1 ≥ 0.2 且检测到手 | 同上 |
| 词元接纳阈值 | 置信度 ≥ **0.5** | `MainViewModel.kt` |
| 断句 | 最后一次有手后 **3 秒** | `MainViewModel.kt` |
| 分类器 | `classifier.tflite`（不是 Keras 模型） | assets |

## 运行

需要用 uv 起一个 Python 3.12 环境（本机默认 3.14 没有 MediaPipe wheel）：

```powershell
# 只做本地识别，不调云
uv run --python 3.12 --with mediapipe --with ai-edge-litert --with numpy `
  python sign2speak-PC-version/recognize_video.py "testdata/videos/xxx.mp4" --pace fast --no-remote

# 本地识别 + 云端组句
$env:S2S_TOKEN = "<token>"
uv run --python 3.12 --with mediapipe --with ai-edge-litert --with numpy `
  python sign2speak-PC-version/recognize_video.py "testdata/videos/xxx.mp4" `
  --pace fast --remote-url "http://127.0.0.1:8080/translate"
```

安全组没放行时，用 SSH 隧道：

```bash
ssh -N -L 8080:127.0.0.1:8080 sign2speak
```

## 输出

每次运行写到 `testdata/results/<视频名>_<时间戳>/`：

- `events.ndjson`：逐次推理的 label/置信度/是否有手，以及词元与句子事件
- `summary.json`：帧数、处理帧率、检测耗时、稳定标签时间线、句子与服务器耗时

## 注意

- `--pace real` 按视频时间轴节流，但本机 CPU 处理约 **4.8 fps**，达不到 30 fps，实际会一直落后；离线分析用 `--pace fast` 即可，两者都按视频时间计算断句。
- PC 端的时延不能直接和手机比：设备与后端都不同，报告中只作“链路打通/正确性”证据。
- 视频是多人拼接片段，硬切处会打断关键点跟踪，解读结果时要注明。
