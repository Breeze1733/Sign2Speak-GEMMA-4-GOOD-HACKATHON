# Sign2Speak 云端节点 Runbook（P1 组句服务）

服务器：`root@121.41.19.165`（阿里云 ECS，Ubuntu 22.04，4 vCPU / 8 GiB，无 GPU）
服务端口：`8080/tcp`（需在阿里云安全组入方向放行）

## 目录约定

```
/opt/sign2speak/
  app/       # 本仓库 server/ 的部署副本（p1/、deploy/、scripts/）
  models/    # gemma-4-E2B-it.litertlm + MANIFEST.txt
  venv/      # uv 管理的 Python 环境
  logs/      # p1.jsonl（每条请求一行）
  results/   # 服务器侧基线数据
  cache/     # LiteRT-LM cache_dir
```

## 首次部署

```bash
# 本地（仓库根目录）
ssh root@121.41.19.165 'mkdir -p /opt/sign2speak'
rsync -az --delete -e ssh server/ root@121.41.19.165:/opt/sign2speak/app/

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
rsync -az --delete -e ssh server/ root@121.41.19.165:/opt/sign2speak/app/
ssh root@121.41.19.165 systemctl restart sign2speak-p1
```

## 接口

`GET /healthz`（无需鉴权）

```json
{"status":"ok","model_loaded":true,"backend":"cpu","litert_lm_version":"0.19.0",
 "queue_depth":0,"queue_max":8}
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
curl -s localhost:8080/healthz | jq
```

模型加载失败时 `healthz.model_error` 会带原因；`/translate` 在加载完成前返回 503。

## 注意

- 手机端 Gemma 走 GPU 后端，本服务器走 CPU 后端；这是"不同后端配置"，写报告时要单独说明，不能当成纯硬件对照。
- 模型与 `litert-lm-api` 版本必须与 Android 端一致（0.19.0），升级任一端都要同步记录。
