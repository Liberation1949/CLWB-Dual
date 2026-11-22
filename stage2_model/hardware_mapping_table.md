# 数学符号 ↔ 真机物理映射与四道硬卡点审计表（Model v1.0）

> **文档状态**：只读冻结底座配套审计文件（Stage 2 · Step 2.1）  
> **物理平台对标**：5G SA / Wi-Fi 6/7 共享上行接入网 + Linux 实时边缘服务器集群（`PREEMPT_RT` / `SCHED_DEADLINE` / `cgroup v2`）  
> **核心目标**：确保模型中每一个数学符号、状态转移与决策动作在真实硬件部署时具备**可观测、可执行、可硬约束保底、不依赖上帝视角**的 1:1 映射关系，支撑后续真机实验一次打通。

---

## 一、核心符号与物理量真机映射总表

| 数学符号 | 模型含义与代码字段 | 真机探针 / 系统接口（API） | 采样触发时机 | 观测滞后 $\delta_{\mathrm{obs}}$ | 测量噪声 / 标定区间 | 因果合法性 |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| $t_k$ | 离散控制周期触发时刻（`obs["time"]`） | `clock_gettime(CLOCK_MONOTONIC)` / RAN MAC TTI 对齐定时器 | 周期 $\Delta t = 50\text{ ms}$ 触发 | $< 50\,\mu\text{s}$ | 内核高精度时钟抖动 $< 0.1\text{ ms}$ | 严格因果 |
| $\Delta t$ | 控制周期步长（`tick_s = 0.05`） | 边缘近实时控制器轮询周期（O-RAN Near-RT RIC 10–100 ms 区间） | 静态配置参数 | $0$ | $\Delta t = 50\text{ ms}$ | 静态配置 |
| $L(t_k)$ | 控制器单次求解墙钟耗时（`wall_s`） | 控制器进程调用前后 `clock_gettime` 差值 | 每次调用求解器后 | 实时产生 | 随待决批次规模 $P$ 超线性或线性变化 | **影响 $t_{\mathrm{exec}}$ 并触发跳拍** |
| $\delta_{\mathrm{sig}}$ | 控制信令下达至无线与算力面生效的固定传输/解析开销（`delta_sig_s = 1.0 ms`） | 下行 PDCCH/MAC CE 授予 + gRPC 派发 RTT 微基准 | 静态/实测标定 | $1.0\text{ ms}$ | $0.5\text{–}1.5\text{ ms}$ 控制信令往返 | 刚性写入状态转移 |
| $t_{\mathrm{exec}}$ | 批决策真实生效时刻（$t_k + L(t_k) + \delta_{\mathrm{sig}}$） | 终端 NIC 入队与执行线程激活时间点 | 动作执行瞬间 | — | 由 $L(t_k)+\delta_{\mathrm{sig}}$ 唯一确定 | **严禁先于 $a_r$ 生效** |
| $\mathcal{P}(t_k)$ | $t_k$ 时刻已到达且未提交的待决任务集（`obs["pending"]`） | 边缘接入网关 / 本地代理上报的任务描述符（到达时刻 $a_r \le t_k$） | 任务到达时异步上报 | $< 1\text{ ms}$ | 零未到达任务泄露（严格过滤 $a_r > t_k$） | 严格因果（零未来任务） |
| $D_r, C_r, d_r$ | 任务 $r$ 输入比特数、计算周期数、绝对截止期（`D_bits`, `C_cycles`, `deadline_s`） | 请求头元信息：`Content-Length` / 算力画像剖析器（FLOP/Cycle 标定） | 任务提交时附带 | $0$ | 标定区间：$D_r \in [60, 120]\text{ KB}$，$C_r \in [10^8, 10^9]\text{ cycles}$，$T_{\max} \in [0.6, 1.0]\text{ s}$ | 提交即已知 |
| $\tau_r^{\mathrm{dn}}$ | 下行结果回传确定性时延（`tau_dn_s`） | 独立 FDD 下行/专用回传链路的包大小除以基站下行速率（$B^{\mathrm{out}}_r / R^{\mathrm{dn}}_{i,j^*}$） | 任务元数据随路携带 | $0$ | $B^{\mathrm{out}}_r \in [0.1, 10]\text{ KB}$，对应 $\tau_r^{\mathrm{dn}} \sim 0.05\text{–}1.2\text{ ms}$ | 确定性小回传 |
| $\mathcal{U}(t_k)$ | 当前在空口实时传输的活动上行流集合（`obs["active_uplinks"]`） | MAC 调度器维护的已激活上行流表（含设备、目标基站、发射功率与剩余字节） | 周期开始快照 | $\le 1\text{ ms}$ | **仅暴露 `rem_bits` 与已发射信息；严禁暴露未来 `end_s`** | 严格去除了未来完成时间泄露 |
| $\mathrm{rem\_bits}_r(t_k)$ | 在传流 $r$ 的剩余未传比特数（`rem_bits`） | PHY/RLC ACK 累计字节反推或终端 BSR（Buffer Status Report） | $\le 5\text{ ms}$ RLC ACK/BSR | $1\text{–}5\text{ ms}$ | BSR 量化误差 $< 2\%$ | 仅依赖历史已交付积分 |
| $\hat{h}_{i,j}$ | 终端 $i$ 到服务器 $j$ 基站天线的估计上行信道增益（`h_hat_dev_ms`） | `nl80211` / SRS 导频参考信号接收功率 RSRP 反算大尺度路径损耗 | $10\text{–}50\text{ ms}$ 周期测量 | $5\text{–}10\text{ ms}$ | 真实信道 $h_{i,j} = g_0 / \max(d_{i,j}, 10\text{ m})^4$；支持 $\sigma_{\mathrm{dB}} \ge 0$ 对数正态 CSI 误差注入 | 可配置真实估计误差 |
| $B,\,N_0,\,p_i$ | 信道带宽、热噪声功率、终端固定发射功率（`bandwidth_hz`, `noise_w`, `p_tx_w`） | 频谱配置（$10\text{ MHz}$）、热噪声基底（$10^{-13}\text{ W}$）、UE RF TPC 标称功率（$0.8\text{–}1.3\text{ W}$） | 静态/连接建立获取 | $0$ | $\alpha$-功率固定单频/多子带配置 | 物理实参 |
| $\mathrm{subband}_i$ | 终端所属正交子带索引（`devices[i]["subband"]`） | 接入侧频域资源块组 RBG / BWP 分配索引（$K \in \{1, 2, 4\}$） | 连接建立时配置 | $0$ | 同子带互相产生同频干扰，跨子带正交隔离 | 物理资源映射 |
| $\text{NIC\_busy}_i,\;q_i$ | 半双工射频占用态与本机排队字节（`radio_busy`, `tx_queue_bits`） | 终端网络协议栈 `TIOCOUTQ` + NIC 驱动 TX 队列深度 | 周期心跳快照 | $< 1\text{ ms}$ | 单终端任意时刻至多 1 条上行链激活（物理硬约束） | 严格物理单工/半双工 |
| $F_j^{\max},\,\rho_{\mathrm{sys}},\,F_j^{\mathrm{avail}}$ | 服务器峰值算力、RT 预留保护上限因子与可用实时算力池（`f_max_hz`, `rho_sys=0.95`, `f_avail_hz`） | `/sys/devices/system/cpu` + Linux 内核参数 `kernel.sched_rt_runtime_us = 950000`（95% RT 封顶） | 启动时读取 cgroup/sysctl | $0$ | $\rho_{\mathrm{sys}} = 0.95$ 为真实操作系统防止内核/IO 饿死的硬限 | **绝对硬件不可逾越边界** |
| $\mathrm{CPU\_used}_j,\,\mathrm{CPU\_res}_j$ | 服务器实时已占用算力与已承诺预留份额（`cpu_used_hz`, `cpu_reserved_hz`） | `cgroup v2 cpu.stat` / `SCHED_DEADLINE` 准入控制账本 | 每次准入与任务释放更新 | $< 0.2\text{ ms}$ | 物理执行总核数/频率永不超过 $F_j^{\mathrm{avail}}$ | 运行时断言硬校验 |

---

## 二、决策动作与真机执行原语映射表

| 数学动作 $a_{r}(t_k)$ | Schema 表达 | 真机执行原语（Edge + Device OS） | 控制面/执行面并发关系 | 资源占用与硬失败语义 |
| :--- | :--- | :--- | :--- | :--- |
| **`defer`**（战略留白） | `{"task": id, "mode": "defer"}` 或缺省 | 任务留在控制器待决队列，不向终端与服务器下发任何执行原语 | 零阻塞、零空口/算力消耗 | 下一调度周期 $t_{k+1}$ 继续暴露（除非 $d_r < t_{k+1}$ 自然过期超时） |
| **`edge`**（批一致卸载） | `{"task": id, "mode": "edge", "server": j, "f_hz": f}` | (1) 在 `t_exec` 向终端下发上行射频激活令牌，进入半双工 NIC 队列按 EDF 排队发射；(2) 数据收齐后进入服务器 $j$ `SCHED_DEADLINE` 运行带宽槽位 $f$ 执行；(3) 执行毕经 FDD 下行返回 | 控制面求解 $L(t_k)$ 与已有在空/算力任务并行推进；**但单线程控制面若 $L(t_k)+\delta_{\mathrm{sig}} > \Delta t$，则阻塞并跳过后续整数拍** | 若在传被同频干扰/排队拖垮导致 $t_{\mathrm{done}} > d_r$，在 $d_r$ 时刻被内核看门狗强制 kill 并记录已耗芯片周期（`wasted_srv_cycles`） |
| **`local`**（终端本地计算） | `{"task": id, "mode": "local", "f_hz": f}` | 在 `t_exec` 注入终端本地 `SCHED_DEADLINE` 队列，占份额 $f \le F_i^{\mathrm{loc}}$，不占用无线信道 | 不与无线传输争夺空口，但受限于终端弱算力（$0.7\text{–}0.9\text{ GHz}$）与排队串行 | $d_r$ 到期未完直接 kill |
| **`drop`**（主动丢弃） | `{"task": id, "mode": "drop"}` | 在 `t_exec` 向客户端返回 `HTTP 503 / Admission Reject`，立刻回收上下文 | 永久脱离队列 | 计入非按时完成（永久失约，不可反悔） |

---

## 三、面向“真机测试一次通过”的四道硬卡点核查结论

1. **卡点 1 · 可观测性零上帝视角（Zero God-View）**：  
   `EdgeUplinkEnv.observe(t_k)` 已从代码接口上物理阻断 `end_s`、`t_tx_end_s` 与未到达任务（$a_r > t_k$）的所有读取通道；第 2 门禁实测验证 `no_end_s_leak = True`、`no_future_task_leak = True`。
2. **卡点 2 · 控制面非零耗时与信令生效延迟（Action & Overhead Fidelity）**：  
   每次调度均按真实测量/设定生效时间 `t_exec = t_k + L(t_k) + delta_sig`（`delta_sig = 1.0 ms`）调度执行；当大突发导致 `SFA` 或 `IBR_5` 决策墙钟耗时超过 $\Delta t = 50\text{ ms}$ 时，仿真器物理模拟单线程控制面阻塞并计入 `skipped_ticks`（见 `stage2_verification_report.json` 实测：`IBR_5` 在 $P=70$ 下 `p99` 远超 $\Delta t$ 达数秒级并触发数百次跳拍）。
3. **卡点 3 · 真机硬件硬边界不可违约（Hard Feasibility Fidelity）**：  
   所有执行资源（终端 NIC 半双工、服务器 `0.95 * F_ms_max` RT 算力封顶、终端 `F_loc` 算力封顶、流体比特守恒误差 $\le 10^{-8}$、因果先后律 $a_r \le t_{\mathrm{exec}} \le t_{\mathrm{tx,s}} < t_{\mathrm{tx,e}} \le t_{\mathrm{cmp,s}}$）全部实现为**事件级物理排队 + 每次事件后强制 `AssertionError` 断言校验**（已累计通过 $> 32,000$ 次全过程状态核验，绝对最大比特守恒误差 $< 10^{-13}$，非法动作 `0`，精确计数实时归档于 [stage2_verification_report.json](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/stage2_verification_report.json)）。
4. **卡点 4 · 实测参数标定与物理尺度校验（Empirical Calibration）**：  
   频宽（`10 MHz`）、载噪比与路径损耗指数（$\sigma = 4.0$，$d_{\min} = 10\text{ m}$ 遵循 3GPP 最小耦合距离）、终端功率（`0.8–1.3 W`）、边缘异构算力及负载规模均直接对齐第一阶段标定实测资产；无量纲检验表明上行批间耦合强度 $\kappa_{\mathrm{UL}} \in [0.343, 9.838]$ 远高于伪瓶颈阈值（$10^{-2}$）。
