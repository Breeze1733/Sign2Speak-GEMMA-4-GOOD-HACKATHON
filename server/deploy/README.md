# Sign2Speak 云端节点 Runbook（P1 组句 + P2 分类/组句）

服务器：`root@121.41.19.165`（阿里云 ECS，Ubuntu 22.04，4 vCPU / 8 GiB，无 GPU）
服务端口：`8080/tcp`（需在阿里云安全组入方向放行）

## 目录约定

```
/opt/sign2speak/
  app/       # 本仓库 server/ 的部署副本（p1/、deploy/、scripts/）
  models/    # gemma-4-E2B-it.litertlm + classifier.tflite + model_meta.json + MANIFEST.txt
  venv/      # uv 管理的 Python 环境
  logs/      # p1.jsonl + p2.jsonl（每条请求一行）
  results/   # 服务器侧基线数据
  cache/     # LiteRT-LM cache_dir
```

## 首次部署

```bash
# 本地（仓库根目录）
ssh root@121.41.19.165 'mkdir -p /opt/sign2speak'
rsync -az --delete --exclude .env -e ssh server/ root@121.41.19.165:/opt/sign2speak/app/

# 服务器
bash /opt/sign2speak/app/scripts/install_env.sh      # 幂等
bash /opt/sign2speak/app/scripts/download_model.sh   # 2.58 GB
cp /opt/sign2speak/app/.env.example /opt/sign2speak/app/.env
vim /opt/sign2speak/app/.env                          # 填 S2S_TOKEN
install -m644 /opt/sign2speak/app/deploy/systemd/sign2speak-p1.service \
  /etc/systemd/system/sign2speak-p1.service
systemctl daemon-reload
systemctl enable --now sign2speak-p1
```

## 更新代码

```bash
rsync -az --delete --exclude .env -e ssh server/ root@121.41.19.165:/opt/sign2speak/app/
ssh root@121.41.19.165 systemctl restart sign2speak-p1
```

## P2 分类器部署

分类器与 Gemma 都放 `/opt/sign2speak/models/`，不要放进 `app/`，否则 `rsync --delete` 会清掉。

```bash
# 本地
scp On_device/app/src/main/assets/classifier.tflite \
    On_device/app/src/main/assets/model_meta.json \
    root@121.41.19.165:/opt/sign2speak/models/

# 服务器：安装依赖后跑强制门禁
/opt/sign2speak/venv/bin/python /opt/sign2speak/app/scripts/check_classifier.py \
  --model /opt/sign2speak/models/classifier.tflite \
  --meta /opt/sign2speak/models/model_meta.json
systemctl restart sign2speak-p1
curl -s localhost:8080/healthz | jq '{classifier_loaded,classifier_runtime,classifier_version,classifier_input_shape,classifier_output_shape,classifier_warmup_ms}'
```

`check_classifier.py` 必须退出 0：它验证 `ai-edge-litert` 能否加载带 `SELECT_TF_OPS` 的模型、输入 `(1,64,708)`、输出 `(1,50)`。失败时把 `server/p1/requirements.txt` 的 `ai-edge-litert` 换成 `tensorflow-cpu==2.16.1`，并在 `.env` 设 `CLASSIFIER_RUNTIME=tf`。

P2 环境变量见 `.env.example`：

- `P2_ENABLED=1` 开启 `/recognize`；设 `0` 可只保留 P1。
- `CLASSIFIER_PATH/CLASSIFIER_META/CLASSIFIER_VERSION/CLASSIFIER_THREADS=1/CLASSIFIER_RUNTIME=auto`。
- `P2_LOG_PATH/P2_QUEUE_MAX/P2_TIMEOUT_S/P2_SESSION_TTL_S/P2_SESSION_MAX/P2_MAX_BODY_BYTES`。
- `P2_DUMP_WINDOWS=1` 会把原始窗口落到 `P2_DUMP_DIR`，用于与 PC harness 对拍。
- `P2_STRICT_VERSION=1` 时 `X-S2S-Model-Version` 不匹配返回 409。

协议、状态码、会话顺序和计时口径见 `server/docs/P2_PROTOCOL.md`。`p2.jsonl` 与 `p1.jsonl` 分开，字段包含 `bytes`、`body_sha256`、`vote_len` 和 `queue/decode/preprocess/inference/vote/total`。

## 接口

`GET /healthz`（无需鉴权）

```json
{"status":"ok","model_loaded":true,"backend":"cpu","litert_lm_version":"0.19.0",
 "queue_depth":0,"queue_max":8,
 "classifier_loaded":true,"classifier_runtime":"ai-edge-litert",
 "classifier_version":"islr-conv1d-transformer-v1",
 "classifier_input_shape":[1,64,708],"classifier_output_shape":[1,50],
 "classifier_warmup_ms":5.03,"classifier_queue_depth":0,"classifier_sessions":0}
```

`POST /translate`，头 `Authorization: Bearer <S2S_TOKEN>`

```json
{"request_id":"t1","session_id":"s1","sentence_index":0,
 "glosses":["WATER","PLEASE"],"style":"friendly","model_version":"gemma-4-E2B-it.litertlm"}
```

返回：

```json
{"request_id":"t1","session_id":"s1","sentence_index":0,
 "sentence":"Can I have some water, please?","status":"generated",
 "queue_ms":3.1,"inference_ms":4123.8,"total_ms":4126.9,
 "model_version":"gemma-4-E2B-it.litertlm","litert_lm_version":"0.19.0"}
```

状态码：`200` 成功；`401` token 不对；`422` 词元为空；`503` 模型加载中或队列满；`504` 推理超时；`502` 生成异常（`status` 里带原因）。

## 计时口径

- `queue_ms`：入队 → worker 开始推理
- `inference_ms`：推理开始 → 文本返回
- `total_ms`：两者之和；不含手机侧网络 RTT 与约 3 秒自动断句等待
- 服务端用 `time.perf_counter()` 单调时钟

## 排障

```bash
systemctl status sign2speak-p1
journalctl -u sign2speak-p1 -n 100 --no-pager
tail -f /opt/sign2speak/logs/p1.jsonl
tail -f /opt/sign2speak/logs/p2.jsonl
curl -s localhost:8080/healthz | jq
```

模型加载失败时 `healthz.model_error` 会带原因；`/translate` 在加载完成前返回 503。分类器失败时 `healthz.classifier_error` 会带原因；`/recognize` 返回 503，但 `/translate` 不受影响。

## 注意

- 手机端 Gemma 走 GPU 后端，本服务器走 CPU 后端；这是"不同后端配置"，写报告时要单独说明，不能当成纯硬件对照。
- 模型与 `litert-lm-api` 版本必须与 Android 端一致（0.19.0），升级任一端都要同步记录。
