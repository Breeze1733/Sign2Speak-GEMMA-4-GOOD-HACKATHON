# Sign2Speak 真机性能测试

**Sign2Speak Timing** 记录手部首次检出、词语识别和句子生成的时间。通过 `-Psign2speakTiming=true` 构建，包名为 `com.sign2speak.app.perf`，可与原版并排安装。手势分类和句子生成均在手机本地运行。

## 安装已有 APK

收到组内已有的测试版 APK 后，在 **Android 10 及以上、ARM64** 的手机上直接打开安装。如果手机提示来源权限，请允许用于打开 APK 的应用安装。

打开 **Sign2Speak Timing** 并允许相机权限。首次使用时点击 **Download Gemma model**，建议用 Wi-Fi 下载约 **2.58 GB** 的模型。已有同名模型文件时，也可以先安装测试版，再用 ADB 将模型复制到测试应用目录。

```sh
adb push gemma-4-E2B-it.litertlm /sdcard/Android/data/com.sign2speak.app.perf/files/
```

## 按需从源码构建 APK

手上没有测试 APK，或需要测试代码的新改动时，可以从本 PR 构建。

电脑需要 JDK 17 或 21，以及 Android SDK Platform 36。建议用 Android Studio 打开 `On_device` 配置 SDK，也可以设置 `ANDROID_HOME`。Windows 下请把仓库放在纯英文路径，例如 `C:/dev/Sign2Speak`，以通过 Android Gradle 插件的路径检查。

在仓库的 `On_device` 目录运行以下命令。

```powershell
.\gradlew.bat :app:assembleDebug -Psign2speakTiming=true
```

macOS 和 Linux 使用 `bash ./gradlew`。APK 生成于 `On_device/app/build/outputs/apk/debug/app-debug.apk`，可以直接发给组员。

## 采集日志

电脑安装 Node.js 20 或更新版本以及 Android SDK Platform-Tools，将 `adb` 加入 PATH。用 USB 数据线连接手机，开启 USB 调试并确认手机上的授权。用 `adb devices` 确认设备显示为 `device`。

在 `On_device` 目录启动采集。

```sh
node performance-tools/collect.mjs --seconds 600
```

有多台设备时加 `--device SERIAL`，也可用 `--adb PATH` 指定 ADB 路径。采集到时自动结束，也可按 Ctrl+C 结束。请等最后一句返回再停止。结果默认保存在用户主目录的 `Sign2SpeakTiming/results/`，可用 `--out DIR` 指定其他目录。

## 测试步骤

1. 等模型加载完成，先试做一轮。正式测试记录手机型号、Android 版本、应用构建、模型版本、网络类型和手势顺序。对比不同版本时，使用同一手机和手势顺序，并保持拍摄条件和网络设置一致。
2. 每轮点一次 **清空并开始新一轮测量**。单词测试每轮只输入一个词。多词测试每轮按预先定好的顺序输入一组词，中间保持同一轮测试。
3. 每个词从双手离开画面开始，进画面时摆好对应手势。识别后移开双手约 0.3–1 秒，再做下一词。日志以至少 200 ms 未检测到手作为新词的分界。
4. 最后一个词做完后保持双手离开画面，约 3 秒后软件自动提交组句。等句子返回再开始下一轮。复测可按单词和多词各 20 轮安排。

## 计时和统计

- **词语识别**。从首次检测到手开始，到软件第一次输出置信指标达到 50% 的词为止。每次手进入最多记录一个词，分别统计每句首词和后续词。提前记下目标词，用来核对识别结果。
- **单词测试**。一次手进入出现多个达标词时，取第一个达标词的时间，其余输出不重复计数。
- **后续词切换**。如果同一次进手内两个不同达标词的输出间隔小于 100 ms，取后一个词，沿用手首次检出的计时起点。例如 512 ms 输出 `girl`，88 ms 后输出 `food`，该词记为 `food`、600 ms。
- **后续词的过早输出**。如果后续词的首个达标输出距进手小于 100 ms，按测试约定剔除该输出，取同一次进手中距起点至少 100 ms 后第一个不同的达标词。没有可替代的词时，只从词语时延统计中剔除这条记录，保留本轮和组句记录。
- **句子生成**。从句子提交到生成文本返回，以 `submit_to_text_ms` 为准。约 3 秒自动断句等待和语音播放另计。只把 `status=generated` 的记录计入成功生成的耗时。
- **汇总**。分别给出单词识别、多词首词、多词后续词和句子生成的最快、最慢、平均耗时，单位为 ms。各时间差用手机端单调时钟计算。

采集器输出原始的 `words.csv` 和 `sentences.csv`。如需按切换规则复核，查 `events.ndjson` 中同一 `session`、`trial_id` 和 `episode_id` 的 `recognition_result`，以 `word_started.hand_seen_ns` 为起点。`words.csv` 的 `hand_detected_to_50pct_ms` 给出首次达标的原始耗时。
