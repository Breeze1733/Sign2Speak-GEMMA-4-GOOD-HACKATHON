# testdata — 视频/帧包测试素材

队友发来的测试素材放这里。媒体文件已被 `testdata/.gitignore` 忽略，不会进 git；只有本 README 和 `manifest.csv` 会被跟踪。

## 目录

```
testdata/
  videos/    # 原始视频：mp4 / mov / avi / mkv
  packs/     # 帧包：pack_frames.py 生成的 S2SF .bin（给 Android 文件导入用）
  manifest.csv  # 片段清单 + 真值标签
```

## 命名

建议 `videos/<词或句>_<编号>.mp4`，`packs/<同名>.bin`，例如：

```
videos/water_please_t01.mp4
packs/water_please_t01.bin
```

一个词一个片段最好；多词句片段结尾要留约 3 秒“手离开画面”，否则 App 的自动断句不触发。

## manifest.csv 填写

| 列 | 含义 |
|---|---|
| clip_id | 唯一编号，如 `water_please_t01` |
| video | `videos/` 下的文件名 |
| pack | `packs/` 下的文件名，没有就留空 |
| signs | 真值词元，空格分隔，如 `WATER PLEASE` |
| style | 期望组句风格 `friendly`/`professional`/`simple` |
| fps | 视频帧率 |
| resolution | 如 `640x480` |
| notes | 拍摄距离、镜像、光照等 |

## 校验

`.bin` 开头应是 ASCII `S2SF`，紧随版本、宽、高、帧数、fps（小端 int32）。放好后我可以直接读头校验它和视频的 fps/分辨率是否一致——不一致说明帧包可能是另一版视频打的。

## 传到手机（文件导入模式）

App 走系统文件选择器，把 `.bin` 放到手机可访问的位置即可：

```bash
adb push testdata/packs/water_please_t01.bin /sdcard/Download/
```

然后在 App 里选择本地文件回放。
