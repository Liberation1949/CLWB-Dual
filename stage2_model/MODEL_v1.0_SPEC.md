# Stage 2 形式化数学建模与本征性质分析总纲（Model v1.0）

> **只读冻结底座规范** · Version: `v1.0` · Status: **Frozen Baseline Specification**  
> **配套可执行底座**：[model_spec.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/model_spec.py) · [exact_oracle.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/exact_oracle.py) · [baselines_stage2.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/baselines_stage2.py) · [hardware_mapping_table.md](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/hardware_mapping_table.md)  
> **严格纪律**：本文档与配套代码仅定义纯物理与数学问题 $\mathcal{P}_{\mathrm{ECU}}$，开展结构解剖与数学杠杆提炼，**严禁提前夹带、命名或构造任何新算法或启发式求解策略**。

---

## Step 2.1 物理真实性框定与最小完备形式化（问题 $\mathcal{P}_{\mathrm{ECU}}$）

### 1. 扩展 Graham 四元组定义 $(\alpha \mid \beta \mid \gamma;\, \mathcal{F}_{t_k},\, \mathcal{A}(s_{t_k}))$

将共享上行无线边缘网络中的在线实时批量卸载与跨层资源控制问题形式化定义为 **$\mathcal{P}_{\mathrm{ECU}}$（Endogenous-Budget Coupled Wireless Edge Uplink-Compute Scheduling）**：

1. **资源与对象拓扑（$\alpha$）**：
   - 终端集合 $\mathcal{N} = \{1,\dots,N\}$：终端 $i$ 配备单条半双工上行射频网卡（任意时刻 $t$ 最多激活 $1$ 条传输链路，同机多任务以 EDF 顺序排队）、固定发射功率 $p_i > 0$、所属子带索引 $\mathrm{sb}(i) \in \{0,\dots,K-1\}$ 及本地 CPU 容量 $F_i^{\mathrm{loc}} > 0$。
   - 边缘服务器集合 $\mathcal{M} = \{1,\dots,M\}$：服务器 $j$ 具有物理峰值算力 $F_j^{\max}$ 与 Linux 实时调度硬上限可用算力 $F_j^{\mathrm{avail}} = \rho_{\mathrm{sys}} F_j^{\max}$（$\rho_{\mathrm{sys}} = 0.95$）。
   - 上行无线链路矩阵 $\mathbf{H} = (h_{i,j})_{N \times M}$，遵循实测路径损耗 $h_{i,j} = g_0 / \max(d_{i,j}, d_{\min})^{\sigma}$（$\sigma = 4.0$，$d_{\min} = 10\text{ m}$）。控制面以周期 $\Delta t = 50\text{ ms}$ 在离散触发点 $t_k = k \Delta t$ 运行决策。

2. **约束分层三清单（$\beta$）**：
   - **清单 A · 核心主导耦合约束（严格正交为 3 维）**：
     1. *连续流体同频干扰与半双工 NIC 排队*：在任一时刻 $t$，设全网活跃在传流集合为 $\mathcal{U}(t)$，其中每个设备 $i$ 在任意时刻最多只有一条链路在传（其余已授予的同设备卸载任务按 EDF 在本地 NIC 发送队列排队等待，队列释放即时切换）。活跃流 $r$（设备 $i=i(r)$，目标服务器 $j=j(r)$）的瞬时香农上行速率受同一子带内并发设备共同干扰：
        $$R_r(\mathcal{U}(t)) = B \log_2 \!\left(1 + \frac{p_{i(r)} h_{i(r), j(r)}}{N_0 + \sum_{u \in \mathcal{U}(t) \setminus \{r\}:\, \mathrm{sb}(i(u)) = \mathrm{sb}(i(r))} p_{i(u)} h_{i(u), j(r)}}\right)$$
        其剩余未传比特 $\mathrm{rem}_r(t)$ 随 $\mathcal{U}(t)$ 每次加入、完工或超时退出实时重算积分：$\frac{\mathrm{d}\,\mathrm{rem}_r(t)}{\mathrm{d}t} = -R_r(\mathcal{U}(t))$。
     2. *内生时间窗缩水与算力份额非线性反演*：批授予通过延长 $t_r^{\mathrm{tx,e}}$ 直接压缩可用计算时间窗 $\text{slack}_r^{\mathrm{cmp}} = d_r - t_r^{\mathrm{tx,e}} - \tau_r^{\mathrm{dn}}$，决定最低可行算力份额下界：
        $$f_r^{\min}(\mathcal{U}, \mathcal{S}) = \frac{C_r}{d_r - t_r^{\mathrm{tx,e}}(\mathcal{U}, \mathcal{S}) - \tau_r^{\mathrm{dn}}}$$
        当 $t_r^{\mathrm{tx,e}}$ 逼近 $d_r - \tau_r^{\mathrm{dn}}$ 时，$f_r^{\min}$ 呈 $1/\text{slack}$ 双曲发散爆炸，极易击穿服务器残余算力容量或令此前已准入的在途任务发生违约。
     3. *控制面非零求解与信令延迟耦合*：在 $t_k$ 决策产生的壁钟计算耗时 $L(t_k)$ 与信令往返延迟 $\delta_{\mathrm{sig}} = 1.0\text{ ms}$ 决定了动作实际激活时刻 $t_{\mathrm{exec}}(t_k) = t_k + L(t_k) + \delta_{\mathrm{sig}}$；且当 $L(t_k)+\delta_{\mathrm{sig}} > \Delta t$ 时，单线程控制面物理阻塞并跳过中间控制周期，直接挤压突发流量生存期。
   - **清单 B · 真机硬边界约束（不可逾越、不可软惩罚化）**：
     - 半双工约束：$\forall t,\,\forall i,\; |\{r \in \mathcal{U}(t) : i(r) = i\}| \le 1$；
     - 硬件算力池硬封顶：$\forall t,\,\forall j,\; \sum_{r \in \mathrm{ComputeSrv}_j(t)} f_r \le \rho_{\mathrm{sys}} F_j^{\max}$；$\forall i,\; \sum_{r \in \mathrm{ComputeLoc}_i(t)} f_r \le F_i^{\mathrm{loc}}$；
     - 份额范围硬限：$f_{\min} = 1\text{ MHz} \le f_r \le F^{\mathrm{avail}}_{j(r)}$（边缘）或 $F_i^{\mathrm{loc}}$（本地）；
     - 严格因果链：$a_r \le t_{\mathrm{exec}}(t_k) \le t_r^{\mathrm{tx,s}} < t_r^{\mathrm{tx,e}} \le t_r^{\mathrm{cmp,s}}$。
   - **清单 C · 显式剥离项清单（含 Stage-1 定量依据）**：
     - 终端电池电量/能量采集约束：阶段一全量日志分析表明，主流标定负载中因电池耗尽触发失败的任务占比不足 $< 0.8\%$，真正主导 33.92% 违约率与 60% 算力浪费的是“同频干扰缩窗 + 控制面过载跳拍”；为保模型 3 维正交性与真机优先对齐 Linux RT/RAN 调度层，显式剥离动态电池模型。
     - 设备间 D2D 协同卸载：在真实基站/边缘路由器控制面架构下，跨终端直连信道 $\mathbf{H}_{\mathrm{dev,dev}}$ 上报开销为 $O(N^2)$ 且真实终端不对外开放直连算力共享池，实测去 D2D 对核心批干扰反噬现象无实质影响，显式剥离。

3. **因果信息流 $(\mathcal{F}_{t_k})$ 与动作空间 $(\mathcal{A}(s_{t_k}))$**：
   - 在决策时刻 $t_k$，因果历史信息集 $\mathcal{F}_{t_k}$ **仅包含**：所有已到达未结待决任务 $(D_r, C_r, d_r, \tau_r^{\mathrm{dn}}, i(r))$、当前正在进行的无线活动流 $(i(r), j(r), p_{i(r)}, \mathrm{rem}_r(t_k), f_r, d_r)$、设备本地队列与可用算力状态、服务器已分配/在算负载、以及可测（含噪声 $\sigma_{\mathrm{dB}}$）信道估计矩阵 $\hat{\mathbf{H}}$。**未来到达流与活动流未来的准确完成时间戳 $t_r^{\mathrm{tx,e}}$ 被工程接口物理屏蔽。**
   - 合法动作空间支持**战略留白（Delay-Allowed）**：对于 $p \in \mathcal{P}(t_k)$，可选择 `defer`（保留待决）、`local`（分配本地算力 $f_r$）、`edge`（指派至服务器 $j$ 并承诺算力配额 $f_r$）或 `drop`（永久拒绝）。

4. **单一原生标量目标（$\gamma$）**：
   $$\max_{\pi \in \Pi_{\mathrm{causal}}} \; J(\pi) \;=\; \frac{1}{R} \sum_{r=1}^{R} \mathbb{I}\!\left(t_r^{\mathrm{done}}(\pi) \le d_r\right)$$
   即严格硬约束下的**全周期按时完成率（On-time Completion Rate）**；不引入人为加权和代理目标。

### 2. 无量纲归一化与伪瓶颈早筛结果

消去秒/赫兹/比特具体量纲后，系统全局行为由 3 个正交无量纲控制参数主导：
1. **上行干扰耦合强度指数**：$\kappa_{\mathrm{UL}} = \dfrac{\mathrm{med}(u^{\mathrm{int}}) - \mathrm{med}(u_0)}{\mathrm{med}(\text{slack}_0)}$（衡量典型半数同子带并发传输引起的时延扩张相对于初始算力松弛窗口的占比）；
2. **归一化系统算力 Offered Load**：$\rho_{\mathrm{cmp}} = \dfrac{\sum_r C_r}{W \left(\sum_i F_i^{\mathrm{loc}} + \sum_j F_j^{\mathrm{avail}}\right)}$；
3. **控制周期相对松弛比**：$\gamma_{\mathrm{tick}} = \dfrac{\Delta t}{\overline{T}_{\max}}$。

对 5 组对齐 Stage-1 的标定场景执行无量纲体检（`Gate 3`）：
- $\kappa_{\mathrm{UL}} \in [0.343,\; 9.838]$（`CAL_BALANCED = 0.357`，`CAL_DEADLINE_TIGHT = 0.343`，`CAL_HETERO_DEV = 0.886`，`CAL_BURST_HEAVY = 0.882`，`CAL_INTERF_DENSE = 9.838`），**远超伪瓶颈判定阈值 $10^{-2}$ 近两个数量级**；
- $\rho_{\mathrm{cmp}} \in [0.605,\; 0.931]$，$\gamma_{\mathrm{tick}} \in [0.050,\; 0.083]$。

---

## Step 2.2 跨年代寻祖投影、精确裁判验证与病灶保真核验

### 1. 1960–2020 基础学科经典谱系与本问题断点

| 时代与经典方法 | 学科谱系与前置核心假设 | 在问题 $\mathcal{P}_{\mathrm{ECU}}$ 中的精确物理/数学断点 |
| :--- | :--- | :--- |
| **`EDF_PreBatch`** *(Liu & Layland 1973; Horn 1974)* | 经典硬实时单机/多机抢占与非抢占调度；假设任务释放后执行时间固定、不随并行任务数恶化 | 使用批前静态干扰计算 $f_r$ 而不感知同批次上行干扰；并发激活导致上传拉长、算力窗口缩水，违约并空烧大量服务器周期 |
| **`LLF_PreBatch`** *(Mok 1983)* | 动态最小松弛优先；假设松弛度估计为外生静态量 | 同一时刻多任务低松弛度涌入反而加剧拥塞坍塌，同样死于批前干扰盲视 |
| **`Bambos_ALP`** *(Bambos-Chen-Pottie, IEEE/ACM ToN 2000)* | 无线蜂窝有源链路保护与渐进移除；采用固定无线 SINR 门限 $\gamma_{\min}$ 控制链路准入与功控 | 仅检查无线侧静态 SINR 阈值，缺乏跨层算力时间窗反演耦合；既会在低 SINR 但紧算力可救的场景误杀吞吐，也会忽略半双工 NIC 队列积压 |
| **`Chen_Game`** *(Chen et al., IEEE/ACM ToN 2016)* | 多用户边缘卸载势博弈 Gauss-Seidel 最优响应；面向平均能效优化，静态同步信道速率 | 忽略连续流体-SINR 动态退出重积分与半双工排队延滞；在紧算力硬约束下会产生博弈策略误差并承受高往返遍历耗时 |
| **`DGOLB`** *(Xu et al., IEEE TMC 2026)* | 边缘计算前沿代表：动态能效最小化模式与服务器容量预留控制 | 在共享信道下按批前状态定价算力份额，高并发触发大规模已准入任务事后超时（Stage-1 核心病灶） |
| **`SFA` / `IBR_5`** *(序贯精确检查与多轮重优化基线)* | 基于全集仿真验证或 Gauss-Seidel 多轮重定价提升静态吞吐 | 每次候选扩展需 $O(P)$ 次流体 rollout 校验，突发规模 $P$ 上升时求解呈 $O(P^2 M)$ 爆炸，决策墙钟耗时突破 $\Delta t = 50\text{ ms}$ 触发控制面跳拍雪崩 |

### 2. 底座执行精度与病灶保真复现（Gate 2, Gate 4, Gate 5 实测结果）

- **全系统精确度与无违约校验**：在 `Gate 2` 全套基线跑测累计 $> 32,000$ 次状态转移断言中（精确运行计数与墙钟时延快照归档于 [stage2_verification_report.json](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/stage2_verification_report.json)），半双工单播、服务器 95% RT 封顶、终端容量、因果先后顺序全部 **零违约**；连续流体积分最大比特守恒相对误差 $< 10^{-13}$（远低于物理阈值 $10^{-8}$）；非法动作数为 `0`。
- **100 组随机微算例精确 Oracle 对拍（`Gate 4`）**：运行全知最优树搜索 `solve_offline_exact`（零调度耗时、全周期全局寻优）对比所有 7 个因果基线：
  - `100 / 100` 实例可行域非空，全知最优平均完成率 `100.0%`；
  - 相对 7 个基线累计 `700` 次比较中，发生基线反超 Oracle 的例数为 **`0`**（均值对比：`EDF_PreBatch = 34.0%`，`LLF_PreBatch = 34.6%`，`DGOLB = 33.1%`，`Bambos_ALP = 75.6%`，`Chen_Game = 85.4%`，`SFA(zero-lag) = 99.2%`，`IBR_5(zero-lag) = 99.2%`）。
- **第一阶段核心失效病灶定量重现（`Gate 5`）**：

| 基线政策 | 共享信道按时完成率 | 共享信道服务器准入失约率 (`miss_rate`) | 共享信道服务器废算率 (`wasted_ratio`) | 正交信道失约率 (`miss_rate`) | 突发 $P=70$ 决策 P99 耗时 | 突发 $P=70$ 控制跳拍 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DGOLB` (TMC 2026)** | **50.7%** | **59.9%** | **49.0%** | **1.6%** | `< 1 ms` | `0` |
| **`EDF_PreBatch` (1973)** | **52.2%** | **62.4%** | **53.7%** | **1.1%** | `< 1 ms` | `0` |
| **`LLF_PreBatch` (1983)** | **56.3%** | **56.4%** | **47.7%** | **1.1%** | `< 2 ms` | `0` |
| **`Bambos_ALP` (2000)** | **~60.0%** | `0.0%` | `0.0%` | `1.1%` | `~3–7 ms` | `0` |
| **`Chen_Game` (2016)** | **~69.5%** | **~33–34%** | **~25–26%** | `~1.6–1.7%` | `~3–7 ms` | `0` |
| **`SFA`** *(受真实墙钟阻塞)* | `< 15%` | `> 89%` | `> 83%` | 亦因阻塞崩溃 | **`>> 50 ms` ($\Delta t$)** | **触发数十次跳拍** |
| **`IBR_5`** *(受真实墙钟阻塞)* | `< 5%` | `100.0%` | `~80.0%` | 亦因阻塞崩溃 | **`>> 50 ms` (数秒级)** | **触发上百次跳拍** |

> **关键证实**：上述对照彻底剥离并重现了问题 $\mathcal{P}_{\mathrm{ECU}}$ 的双重痛点：一方面，一旦剥离无线耦合进入正交信道，`DGOLB` 与 `EDF_PreBatch` 失约率瞬间从 **60%–62%** 跌落至 **1.1%–1.6%**（按时率达 **98.5%–99.2%**），证明“批前盲视导致时间窗反噬”绝非代码或调度巧合，而是纯粹的跨层同频物理现象；另一方面，`SFA` 与 `IBR_5` 在零计算延迟下虽在微观或局部可达较高理论可行性，但在真实控制面耗时下直接引发 P99 数百毫秒至多秒级阻塞与数十至上百次控制跳拍，彻底崩溃。

---

## Step 2.3 纯问题本征数学解剖（严禁提出新算法）

### 1. 两种严格等价数学表示

#### 表示 A · 连续流体-SINR 单调算子与遗传独立系统（Hereditary Independence System）
在固定决策时刻 $t_k$ 与当前活跃链路集合 $\mathcal{U}_0 = \mathcal{U}(t_k)$ 下，考虑任意候选卸载批次及服务器映射 $\mathcal{S} = \{(r, j(r))\}$。由每个活跃设备的干扰贡献只取决于其当前是否在传（$p_{i(u)} h_{i(u), j}$ 与当前服务任务无关），流体传输方程构成一个保序常微分方程组比较系统，诱导出全局上传完成时间映射 $\mathbf{T}^{\mathrm{tx,e}}(\mathcal{S})$ 与最小所需 CPU 算力映射 $\mathbf{f}^{\min}(\mathcal{S})$。

**定理 2.1（全局遗传反单调性 / Downward-Closed Feasibility System）**：  
对于任意固定服务器分配下的两批卸载集合 $\mathcal{S}_1 \subseteq \mathcal{S}_2$：
1. *传输完了时刻保序性*：$\forall r \in \mathcal{U}_0 \cup \mathcal{S}_1,\; t_r^{\mathrm{tx,e}}(\mathcal{S}_1) \le t_r^{\mathrm{tx,e}}(\mathcal{S}_2)$；
2. *算力份额需求保序性*：$\forall r \in \mathcal{S}_1,\; f_r^{\min}(\mathcal{S}_1) \le f_r^{\min}(\mathcal{S}_2)$；
3. *遗传可行性（Downward-Closed Independence）*：记静态切分容量可行域族 $\mathcal{I}(\mathcal{U}_0) = \{\mathcal{S} : \text{满足活性链路保活与 }\sum_{r\in\mathcal{S}_j} f_r^{\min}(\mathcal{S}) \le F_{j,\mathrm{rem}}^{\mathrm{avail}}\}$，则 $\mathcal{S}_2 \in \mathcal{I}(\mathcal{U}_0) \implies \mathcal{S}_1 \in \mathcal{I}(\mathcal{U}_0)$。

*实证核验（Gate 6）*：在 `200` 组跨同频多子带、多在空链路及同机多任务随机混合试验证明，`200 / 200` 全部严格成立（`t_tx_end` 最大数值浮差 $6.94\times 10^{-18}\text{ s}$，`f_min` 最大违逆 $0.0\text{ Hz}$）。这一遗传结构从第一性原理保证了：**一旦从当前超集批次中移除某个阻塞/高干扰任务，所有保留任务与在空链路只能变得更安全，绝不会出现撤回诱导的连锁新违约。**

#### 表示 B · 多周期因果部分可观测马尔可夫决策 / 停时控制（Causal POMDP）
跨多周期 $\{t_0, t_1, \dots\}$，定义状态为物理资源全状态 $s_k$，控制面仅可经由延迟掩码 $\mathcal{F}_{t_k} = \Phi_{\mathrm{obs}}(s_k)$ 观测。合法策略 $\pi$ 是从历史信息过滤流 $\{\mathcal{F}_{t_k}\}$ 到包含 `defer` 的受限动作空间 $\mathcal{A}(\mathcal{F}_{t_k})$ 的映射；每次动作转移受控于内生墙钟计算耗时 $L(\pi, \mathcal{F}_{t_k})$，决定状态向前步进的物理跳拍数 $m_k = \max(1, \lceil (L+\delta_{\mathrm{sig}})/\Delta t \rceil)$，以最大化全视界截止期命中率 $J(\pi)$。

### 2. 四大退化极限锚点（精确解析基准）

当耦合维度的某一项退化至极限时，问题 $\mathcal{P}_{\mathrm{ECU}}$ 退化为以下 4 类拥有精确闭式解或多项式最优解的经典基准问题（已通过 `verify_degenerate_anchors` 全部 4/4 验证与 `solve_single_tick_exact` 一致）：
1. **锚点 1 · 单服务器正交极限（$\kappa_{\mathrm{UL}} = 0, M = 1$）**：链路速率脱耦为常数 $u_r = D_r / R_0(r)$，需求化为静态权重 $f_r^{\min}$，最大准入数化为单位价值背包问题，$\mathbf{f}^{\min}$ 升序排列贪心填充即**严格多项式精确最优**；
2. **锚点 2 · 无限服务器算力极限（$F_j^{\mathrm{avail}} \to \infty$）**：算力份额不再制约任何任务，问题退化为纯无线连续流体 SINR 截止期准入控制；
3. **锚点 3 · 无限信道带宽极限（$B \to \infty$）**：上行时延趋于 $0$，退化为纯计算截止期份额打包问题（相对误差验证 $\le 10^{-5}$）；
4. **锚点 4 · 单任务孤立极限（$R = 1, \mathcal{U}_0 = \emptyset$）**：退化为解析比较本地算力 $F_i^{\mathrm{loc}}$ 与最优边缘链路 $\max_j \{C / (d - t_{\mathrm{exec}} - D/R_0(j) - \tau^{\mathrm{dn}})\}$ 的单步闭式判定。

### 3. 逐约束剥离复杂度跃迁图谱（Constraint-Ablation Complexity Map）

| 约束保留状态 | 问题计算复杂性 | 致难根本机理 |
| :--- | :---: | :--- |
| 剥离无线同频耦合（正交化 $\kappa_{\mathrm{UL}}=0$）且单机 | **多项式时间 $P$**（Moore-Hodgson EDF / 升序背包） | 算力要求 $f_r^{\min}$ 为固定静态常量，无批间副作用 |
| 剥离计算截止期约束（无限算力 $F \to \infty$）保留几何 SINR 准入 | **NP-hard**（Goussevskaia et al. 2007 几何 SINR 链路最大集归约） | 多链路空间干扰图与幂律叠加构成非线性独立集搜索 |
| 单机共享上行（$M=1$）+ 内生 $1/\text{slack}$ 算力份额反演（**本问题核心双约束叠加**） | **强 NP-hard & 非子模最大化** | 1. SINR 容量划分等价于广义干扰独立集；2. 算力需求 $C_r/(d_r - t_r^{\mathrm{tx,e}}(\mathcal{S}) - \tau_r^{\mathrm{dn}})$ 在高耦合区呈超可加（Super-additive）爆炸，导致边际收益严重非单调且破坏标准次模性 |
| 引入时间轴因果动态到达与非零决策时延 $L(t_k)$ | **无竞争比上界（若禁止战略留白或忽略批内耦合）** | 见下文下界定理 $L_1, L_2, L_3$ |

### 4. 三大策略类本征下界 / 不可能性定理及 $N \le 5$ 手算可验反例

在不构造任何新算法的前提下，严格证明前人三类受限范式在本数学底座 $\mathcal{P}_{\mathrm{ECU}}$ 上必然存在结构性性能天花板（经 `exact_oracle.py` 全量数值验证）：

1. **定理 $L_1$（批前盲视策略类 $\Pi_{\mathrm{pre}}$ 的 0% 崩塌下界）**：  
   存在 $N=4, M=1, R=4$ 对称实例（`CE_L1_PREBATCH`：$d_r = 0.160\text{ s},\, D_r = 800\text{ kb},\, C_r = 1.2\times 10^9,\, h_{i,0}=10^{-10},\, F^{\mathrm{avail}} = 38\text{ GHz},\, F^{\mathrm{loc}} = 0.5\text{ GHz}$）。所有仅基于在空流 $\mathcal{U}_0$ 定价的策略（`EDF_PreBatch`, `LLF_PreBatch`, `DGOLB`）在 $t_0=0$ 均误判上行仅需 $8.03\text{ ms}$、算力仅需 $8.00\text{ GHz}$，进而将全部 $4$ 个任务同时放行；物理并发后 $\text{SINR}$ 骤降至 $0.333$（$4.15\text{ Mbps}$），上行拖延至 $193.8\text{ ms} > 160\text{ ms}$，按时完成数严格跌为 **`0 / 4`（0%，违约率 100%）**；而全知精确批一致与跨拍流水线可达 **`3 / 4`（75%）**，绝对间隙达 **`75.0 pp`**。

2. **定理 $L_2$（非延迟提交策略类 $\Pi_{\mathrm{non\text{-}delay}}$ 的严格次优间隙）**：  
   存在 $N=3, M=1, R=3$ 实例（`CE_L2_NONDELAY`：紧约束重上传流 Task 0 在 $t=0$ 启动至 $0.0712\text{ s}$，后续两任务于 $0.045\text{ s}$ 到达并于 $t_1=0.050\text{ s}$ 被暴露）。任何剥夺 `defer` 动作权的非延迟策略，在 $t_1=0.050\text{ s}$ 要么因并发放行新流而将 Task 0 速率从 $99.7\text{ Mbps}$ 压垮至 $9.99\text{ Mbps}$ 杀掉在空任务，要么永久丢弃新到达任务——经 `solve_offline_exact(non_delay=True)` 穷举验证其理论极限严格不超过 **`2 / 3`（66.7%）**（`SFA_NoDefer` 实测 `1 / 3`），而允许战略留白的最优解在 $t_1$ 暂缓并在 $t_2=0.100\text{ s}$ 合并发射，严格达成 **`3 / 3`（100.0%）**，下界分离 **`33.3 pp`**。

3. **定理 $L_3$（外部性盲视近视贪心策略类 $\Pi_{\mathrm{myopic}}$ 的 $3\times$ 吞吐坍塌）**：  
   存在 $N=4, M=1, R=4$ 非对称信道实例（`CE_L3_MYOPIC_EDF`：近端重包 Task 0 截止期早 $10\text{ ms}$ 但信号高达 $2\times 10^{-8}\text{ W}$——强出同组其余 3 任务 $200\times$）。严格按 EDF 先入先出贪心验证可行集的近视策略（如 `SFA`）必先放行合法的 Task 0，其在 $[0.001, 0.1117]\text{ s}$ 霸占信道将同辈压制至 $72\text{ kbps}$ 并挤爆剩余计算窗口，导致 **`SFA = 1 / 4`（25.0%）**、`EDF_PreBatch = 0 / 4`；而跳过/延后极端外部性任务转而协同接纳其余 $3$ 任务达到 **`3 / 4`（75.0%）**，证明全局最优相对近视贪心存在 **`+50.0 pp`（$3\times$）** 本征间隙。

### 5. 二维无量纲相空间区域划分

![2D Dimensionless Phase Space](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/phase_space_2d.png)

基于 $6 \times 6 \times 2$ CRN 扫频网格（$\rho_{\mathrm{cmp}} \in [0.247, 0.989]$，$\kappa_{\mathrm{UL}} \in [0.027, 1.380]$），$(\rho_{\mathrm{cmp}}, \kappa_{\mathrm{UL}})$ 相平面自然分裂为三个物理区：
- **Zone I · 弱耦合退化区（$\kappa_{\mathrm{UL}} \le 0.12$）**：实测网格行 $\kappa_{\mathrm{UL}}=0.027$ 上，`DGOLB` 失约率稳定为 **`0.0%`**、`EDF_PreBatch = 100.0%`，批内可行投影相对批前基线增益严格为 **`0.0 pp`**——此时复杂求解无正收益，切忌过度设计；
- **Zone II · 算力硬饱和区（$\rho_{\mathrm{cmp}} \ge 0.90$）**：算力总量物理供不应求，任何算法均受硬件产能墙截断；
- **Zone III · 核心耦合相变主战区（$\kappa_{\mathrm{UL}} \in [0.20, 1.40],\; \rho_{\mathrm{cmp}} \in [0.35, 0.90]$）**：传统批前基线在此处触发雪崩式 `miss_rate = 33.3%–69.5%`，批一致可行求解在不考虑计算耗时下可释放 **`+22.1 pp` 至 `+43.7 pp`** 的巨大结构红利；但实时求解开销又在此区随 $P$ 呈现相变卡顿——**这正是后续阶段三算法设计的唯一合法主攻相区。**

---

## Step 2.4 数学杠杆提炼与底座只读冻结

### 1. 纯微扰与对偶敏感性剖析（Why Optimal Solutions Work）

从物理方程直接微扰求导，得到支配理想最优解结构的两条解析规律：

1. **跨层微扰放大定律（Cross-Layer Interference-to-Compute Amplification Law）**：  
   对任一边缘任务 $r$ 分配至服务器 $j$，在稳态有效干扰 $I_j$ 下其最低算力需求 $f_r^{\min} = C_r / \text{slack}_r^{\mathrm{cmp}}$ 对额外接入同频干扰增量 $\partial I_j$ 的精确敏感度由链式法则给出：
   $$\frac{\partial f_r^{\min}}{\partial I_j} \;=\; \frac{C_r D_r}{\big(\text{slack}_r^{\mathrm{cmp}}\big)^2 \cdot B \ln 2 \cdot \log_2^2(1+\mathrm{SINR}_{r,j})} \cdot \frac{S_{r,j}}{(N_0 + I_j)(N_0 + I_j + S_{r,j})}$$
   - **揭示机理**：当同频上行链路 $u$ 接入，其注入服务器 $j$ 的物理干涉功率为 $p_u h_{u,j}$；然而这一干扰成本在**跨层计算约束下被 $\big(1/\text{slack}_r^{\mathrm{cmp}}\big)^2$ 二次方剧烈放大**。一个在普通无线侧看似可承受的数分贝 SINR 下降，如果作用在一位已经接近算力时间窗尽头的同批伙伴或紧期限在空链路 $r$ 上，会立即引发后者算力影子价趋近无穷（直接击穿服务器总量或导致死锁）。

2. **边际拥塞外部性与战略留白（`defer` vs `commit`）临界不等式**：  
   设当前服务器可用核余量对算力需求单位的局部拉格朗日对偶乘子（影子价格）为 $\lambda_j \ge 0$，现有在空链路加批内干扰综合时延外溢惩罚乘子为 $\mu_{i,j}^{\mathrm{UL}}$。对于任务 $r$，若在周期 $t_k$ 立即放行可获得单位吞吐价值 $1$，但其产生的**跨层总机会成本（Shadow External Cost）**包含两部分：① 占用服务器算力机会成本 $\lambda_j f_r^{\min}(\mathcal{S} \cup \{r\})$；② 对当前在空流 $\mathcal{U}(t_k)$ 与低截止期同批任务产生的二次方外溢算力挤压 $\sum_{q \in \mathcal{U}(t_k) \cup \mathcal{S}} \lambda_{j(q)} \Delta f_q^{\min}(r)$。  
   只有当留白至 $t_{k+1}$ 仍保留充足窗口（$t_{k+1} + \delta_{\mathrm{sig}} + u_0(r) + \frac{C_r}{F_{j}^{\mathrm{avail}}} + \tau^{\mathrm{dn}}_r \le d_r$）且上述边际外部性成本大于即期吞吐净收益阈值时，**战略性延后 `defer` 才在数学上严格优于强行并发硬塞或永久丢弃**。

---

### 2. 《数学杠杆清单（Mathematical Leverage Inventory）》

为了使后续阶段三的算法设计严格回归“在同一数学底座上做定向求解与局部改进”，明确划分以下两类结构：

| 结构分类 | 具体内容（Stage 2 严格导出） | Stage 3 约束法则 |
| :--- | :--- | :--- |
| **精确可解析骨架** *(Model-Derived Skeleton)* | 1. **连续流体-SINR 遗传独立系统结构**（定理 2.1）：任意不合群或高代价成员的剔除必单调降低留存全集合的传输耗时 $t_r^{\mathrm{tx,e}}$ 与 CPU 需求 $f_r^{\min}$；<br>2. **半双工设备 NIC 严格单流排他与同子带去重律**：同一设备在任一瞬间对其它链路的同频物理辐射至多算 1 份，且该设备自身队列延滞按 EDF 线性串行累加；<br>3. **四项边界精确退化锚点**与**单任务本地 vs 边缘精确切换下界**：所有合法求解器在 $\kappa_{\mathrm{UL}} \to 0$ 必须退化贴近精确解；<br>4. **单线程控制面预算红线**：决策时延 $L(t_k)$ 必须稳定控制在 $\ll \Delta t = 50\text{ ms}$（含并发高突发 $P \ge 70$），否则引发物理跳拍自我摧毁。 | **硬性守恒骨架**：Stage 3 严禁违反遗传单调性前提，也严禁隐藏或绕开壁钟时延惩罚 $t_{\mathrm{exec}} = t_k + L(t_k) + \delta_{\mathrm{sig}}$。 |
| **唯一合法近似切入口** *(Legal Approximation Entry Points)* | 1. **切入口 1（低复杂度批一致映射与外部性定价近似器）**：因全集合流体反复校验（$O(P^2 M)$）在突发期击穿 $50\text{ ms}$ 控制周期，且全局联合选择为强 NP-hard 问题（定理 $L_1, L_3$），Stage 3 允许且**仅允许**在“如何用 $O(P M)$ 或线性级别低复杂度算子逼近流体批一致不动点、并识别高二次方外溢干扰任务”这一环引入结构近似或对偶定价机制；<br>2. **切入口 2（不确定环境下的因果停时与抗抖动安全裕度）**：由于观测信道 $\hat{\mathbf{H}}$ 存在 CSI 估计误差、未来到达未知且战略留白优于非延迟强塞（定理 $L_2$），Stage 3 允许在“何时主动 `defer` 以保护紧期限在途流（停时边界）及如何在 $\hat{\mathbf{H}}$ 误差和 $L(t_k)$ 抖动下构造不过度保守的可验算力/干扰上界包络”处设计估计器或证书近似。 | **唯一创新承载区**：后续阶段三的所有算法构想均须显式声明自己替换或改进的是切入口 1、切入口 2 或二者协同；底座物理公式与评估规范绝对锁定。 |

---

### 3. 显式版本变更与回炉协议（Model Revision Gate）

1. 本文件与 [model_spec.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/model_spec.py)、[exact_oracle.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/exact_oracle.py)、[baselines_stage2.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/baselines_stage2.py)、[hardware_mapping_table.md](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/hardware_mapping_table.md)、[stage2_verification_report.json](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/stage2_verification_report.json) 及 [FREEZE_MANIFEST_v1.0.sha256](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/FREEZE_MANIFEST_v1.0.sha256) 共同签发为 **`Model v1.0` 只读冻结底座**。
2. 进入阶段三（算法设计）与阶段四（证伪质检及真机对照）后，**严禁悄悄修改参数、调松硬约束、消除控制面耗时或回填未观测状态以迎合特定策略分数**。
3. 若真机实验或后续分析暴露了模型未刻画的关键物理假设（例如多普勒快衰落、下行多用户拥塞不可忽略等），必须冻结算法迭代，显式回退至阶段二修订并公开发布 `Model v1.1 / v2.0`，同步重跑全套 8 门禁校验及双轨基线重标定。

---

## 四、Stage 2 准出 8 项硬门禁逐一签收清单

- [x] **1. 最小完备无算法夹带**：已完成四元组与 3 类分层清单定义；AST/正则扫描覆盖 `stage2_model/` 全目录零新算法/启发式术语夹带。
- [x] **2. 真机物理保真闭环**：逐符号完成真机探针映射并零未来信息泄露；全过程通过 $> 32,000$ 次硬件硬约束运行断言，最大比特相对积分误差 $< 10^{-13}$。
- [x] **3. 无量纲安检通过**：标定区间实测 $\kappa_{\mathrm{UL}} \in [0.343, 9.838]$，$\rho_{\mathrm{cmp}} \in [0.605, 0.931]$，$\gamma_{\mathrm{tick}} \in [0.050, 0.083]$，无 $<10^{-2}$ 伪瓶颈。
- [x] **4. 可执行底座与精确裁判就绪**：`100/100` 随机微算例 `solve_offline_exact` 全部可行且严格压制 7 个因果基线（反超 `0` 例，总耗时约 `1 s`）。
- [x] **5. 跨年代寻祖与向上病灶保真 100% 重现**：`DGOLB` 共享失约率 `59.9%` vs 正交 `1.6%`；`IBR_5`（数秒级耗时，上百次跳拍）与 `SFA`（数百至上千毫秒级耗时，数十次跳拍）在 $P=70$ 突发下真实崩溃。
- [x] **6. 等价数学表示与退化锚点完备**：流体遗传反单调性 `200/200` 全通过；4 大退化解析锚点 `4/4` 精确匹配。
- [x] **7. 策略类本征下界与 2D 相空间已锁死**：`L1 / L2 / L3` 三个极小可手算反例（$0/4\text{ vs }3/4$、$\le 2/3\text{ vs }3/3$、$1/4\text{ vs }3/4$）全验通，2D 相图绘制完毕。
- [x] **8. 《数学杠杆清单》落盘且 `Model v1.0` 已冻结**：SHA-256 冻结清单生成完毕，可正式移交阶段三求解。
