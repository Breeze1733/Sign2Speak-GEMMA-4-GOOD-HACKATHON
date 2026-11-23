# 网络条件控制（P1 评测用）

目标是开题报告 8.1 的矩阵：RTT 约 10 / 50 / 100 ms，带宽上限约 1 / 5 / 20 Mbps，
**每次都要实测校准并把实际值写进报告**。

## 为什么不能在手机上做

手机在 4G/5G 上，没有 root 就改不了 `tc`。所以只能在**链路中间**或**服务器出口**做。

## 方案 A：Linux 路由器 / 热点（最干净，推荐）

让手机连一台你能控制的 Linux 机器的热点，在它的出口网卡上做双向整形。

```bash
IFACE=wlan0

# RTT 50ms ≈ 单向 25ms（两端各 25，合计 50）
tc qdisc add dev $IFACE root handle 1: prio
tc qdisc add dev $IFACE parent 1:3 handle 30: netem delay 25ms

# 带宽 5 Mbps（tbf 也可换成 netem rate）
tc qdisc add dev $IFACE parent 1:3 handle 40: tbf rate 5mbit burst 32kbit latency 400ms
```

双向都要整形才算"链路 RTT 50ms"：如果在同一台机器的一个网卡上，`netem delay` 只作用于
该网卡**发出的**方向；手机→服务器方向要在另一侧（或同一网卡的 ingress 用 `ifb`）再做一次。
不确定时，用方案 C 的桌面工具更直观。

**重置**：

```bash
tc qdisc del dev $IFACE root
```

## 方案 B：只在服务器出口整形（简单，但有偏差）

在 `121.41.19.165` 上：

```bash
IFACE=$(ip route get 1.1.1.1 | awk '{print $5; exit}')
tc qdisc add dev $IFACE root netem delay 25ms
tc qdisc add dev $IFACE root tbf rate 5mbit burst 32kbit latency 400ms
# 注意：同一个 root 只能挂一个 qdisc，通常用 prio + 子 qdisc 组合，见方案 A
tc qdisc del dev $IFACE root
```

**偏差**：只影响服务器→手机方向，上行（手机→服务器）不受控，RTT 不对称。
报告里必须写明"仅服务器出口整形"。

## 方案 C：Windows 热点 + 桌面限速工具

手机连电脑热点，用 NetLimiter / Clumsy / TMeter 对热点网卡加延迟和带宽上限。

- 优点：不用 Linux，双向可控（工具按连接整形）。
- 缺点：工具行为不透明，**必须用 ping/iperf3 实测校准**，并记录工具版本和参数。

## 校准（每次改完都要做）

```bash
# RTT：目标 10/50/100ms
ping -c 20 121.41.19.165
# 带宽：目标 1/5/20 Mbps
iperf3 -c 121.41.19.165 -t 10 -R      # 服务器需要跑 iperf3 -s
```

把 `ping` 的 avg 和 `iperf3` 的实际吞吐写进该条件的 `meta.json`，例如：

```json
{"condition": "p1-rtt50-bw5", "rtt_ms_target": 50, "rtt_ms_actual": 52.3,
 "bw_mbps_target": 5, "bw_mbps_actual": 4.8, "shaping": "linux router both directions"}
```

## 实验矩阵

| 条件 | RTT 目标 | 带宽目标 |
|---|---:|---:|
| `p1-rtt10-bw20` | 10 ms | 20 Mbps |
| `p1-rtt50-bw5` | 50 ms | 5 Mbps |
| `p1-rtt100-bw1` | 100 ms | 1 Mbps |

每个条件至少 30 次，顺序随机，记录实际校准值。再加断网/超时作为可靠性场景。
