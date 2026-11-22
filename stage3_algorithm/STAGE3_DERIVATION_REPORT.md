# 阶段三（算法设计阶段）推导与受限演化完整报告（`CLWB-Dual`）

> **当前阶段**：Stage 3（基于冻结 `Model v1.0` 底座的对抗推导、精简重构与五门禁锁定）  
> **只读物理底座**：[MODEL_v1.0_SPEC.md](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/MODEL_v1.0_SPEC.md) · [FREEZE_MANIFEST_v1.0.sha256](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/FREEZE_MANIFEST_v1.0.sha256)（`8/8` SHA-256 哈希零篡改验证通过）  
> **Stage-3 核心代码与验证资产**：[champion_core.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage3_algorithm/champion_core.py) · [common_guard.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage3_algorithm/common_guard.py) · [candidates_routes.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage3_algorithm/candidates_routes.py) · [benchmark_suite_s3.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage3_algorithm/benchmark_suite_s3.py) · [soundness_test.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage3_algorithm/soundness_test.py) · [guard_stage3.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage3_algorithm/guard_stage3.py) · [stage3_results.json](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage3_algorithm/stage3_results.json)

---

## 一、底座只读核验与两阶防显然反例过滤（Step 3.1.1）

### 1.1 Stage-2 冻结底座零篡改核验
`soundness_test.py` 在每次运行的首步即逐字节重算 [FREEZE_MANIFEST_v1.0.sha256](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage2_model/FREEZE_MANIFEST_v1.0.sha256) 所列全部 `8` 个 Stage-2 文件的 SHA-256 摘要，实测 `checked_files = 8, mismatches = []`，确认 `Model v1.0` 的因果观测边界、连续流体-SINR 积分、系统安全计算比例 $\rho_{\mathrm{sys}} = 0.95$ 与真实壁钟时延模型完全未遭改动。

### 1.2 第一阶过滤：`CommonGuard` 与 `OurCore` 严格分界
针对阶段二揭示的工程可修补失效模式，凡属审稿人直观可见的工程常识与一阶静态护栏均已下放至 [common_guard.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage3_algorithm/common_guard.py)，并在主实验与盲测中公平挂载于全部 `7` 个外部对照基线（`+CG`）：
1. **动作合法性与算力封顶清洗**：拒绝越界编号、非法模式及超出设备/服务器实时可用算力上限或低于 $f_{\min}$ 的频率指令；
2. **微秒级标称安全裕度**：在空闲算力裕度内为已准入流补充 `2%` 标称防御余量；
3. **一阶忙 NIC 延后**：当终端射频在忙且当前活跃流孤立剩余耗时加上候选任务孤立发送与满载计算耗时已静态超过截止期时，提前延后（`defer`）；
4. **一阶静态活跃链路尾删保护**：以瞬时并发干扰保守估计活跃流延时，若导致活跃流静态超时，则自尾部逐步撤回新增边缘准入。

> **严格隔离边界**：`CommonGuard` 不含连续流体时间跨层外部性定价、凸二次 Pigouvian-EDF 排序、因果 2 拍流体地平线、`O(log P)` 遗传前缀收缩或拍+上传时机护盾等核心求解算子（`OurCore`）。

### 1.3 第二阶锁定：不可显然修补的二阶耦合反例 `CE_2ndOrder`
在 [candidates_routes.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage3_algorithm/candidates_routes.py#L32-L70) 中构造 $N=5, M=1, R=5$ 的二阶耦合最小反例（`B = 10 MHz`, `N_0 = 1e-13 W`, `F_ms_max = 45 GHz`, `F_avail = 42.75 GHz`, `F_loc = 0.4 GHz`）：

| 任务 | 终端 $i$ | 到达 $a_r$ / 截止 $d_r$ | 载荷 $D_r$ / 算力 $C_r$ | 发射功率与信道增益 $(p_i, h_{i,0})$ | 孤立速率与孤立耗时 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0:0` | $0$ | $0.000\text{ s}$ / $0.105\text{ s}$ | $5.50\text{ Mb}$ / $3.0\times 10^8$ | $(1.0\text{ W},\, 10^{-10})$ | $99.66\text{ Mbps}$ / $55.19\text{ ms}$ |
| `1:0` | $1$ | $0.045\text{ s}$ / $0.216\text{ s}$ | $18.0\text{ Mb}$ / $1.5\times 10^8$ | $(2.0\text{ W},\, 10^{-8})$（**$200\times$ 强干扰体**） | $176.10\text{ Mbps}$ / $102.21\text{ ms}$ |
| `2:0..4:0` | $2,3,4$ | $0.045\text{ s}$ / $0.228\text{ s}$ | $0.40\text{ Mb}$ / $6.5\times 10^8$ | $(1.0\text{ W},\, 10^{-10})$ | $99.66\text{ Mbps}$ / $4.01\text{ ms}$ |

在 $t_0=0.000\text{ s}$ 准入 `0:0`、$t_1=0.050\text{ s}$ `CommonGuard` 因忙 NIC 将新到达任务推迟至 $t_2=0.100\text{ s}$ 后，真正的二阶并发死结于 $t_2$ 引爆：若近视策略按早截止期接纳强干扰重包 `1:0`（$0.216\text{ s} < 0.228\text{ s}$），其在 $[0.101, 0.2032]\text{ s}$ 期间把 3 条同伴速率压低至 $72\text{ kbps}$，使其计算窗压缩至 $\le 0.0198\text{ s}$ 并超出 $42.75\text{ GHz}$ 封顶。实测结果（[stage3_results.json](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage3_algorithm/stage3_results.json)）：`DGOLB+CG = 1/5`、`SFA+CG = 2/5`，而 **Oracle = `4/5`**（延后/弃置 `1:0`、同批准入 `2:0..4:0`），**`CLWB-Dual = 4/5` 严格追平全知最优 Oracle**。

---

## 二、数学杠杆择优与对抗精简重构（Step 3.1.2 & Stage-4 反拼盘物理切除）

### 2.1 三候选数学路线横向对比与强敌倒逼
沿着 Stage-2 的遗传反单调骨架与跨层连续流定价杠杆，在相同底座上横向评测 `RouteA_CLSP`（线性对偶排序）、`RouteB_Waterfill`（流体平滑对数均衡 + 遗传收缩投影）与 `RouteC_Whittle`（延后代价比 Whittle 索引）：`RouteB` 在 2 种子预赛上以 `83.93%`（`miss = 2.05%`）显著击败 `RouteA`（`82.01%`）、`RouteC`（`43.68%`）与 `Chen_Game+CG`（`68.78%`），并在全部 4 组基础与高阶反例（`L1/L2/L3/CE_2ndOrder`）胜出而入选精炼单核。

随后在引入经典无线干扰感知强基线 `Bambos_ALP+CG`（`K=12` 上稳定取得 `89.08%` 且 `0.00%` 失约）及 Stage-4 双轨盲测/反拼盘剪枝审计的联合倒逼下，我们清除了早期演化阶段残余的**四项底层结构缺陷与三项负协同冗余组件**：
1. **物理切除三项负协同/冗余机制**：移除对同子带活跃流重复扣罚的线性项 `active_dmg`（曾引入 `-0.20 pp` 负优化且已被活跃半松弛硬盾完全覆盖）、移除无统计收益的次优服务器填补切换 `alt_srv_only`（`-0.19 pp`），以及移除失真加权代价选机 `srv_cost_pm`（移除备选切机后在分布外场景导致 `-0.73 pp` 负优化），统一回归最优信道增益可行选机与单位阈值（$\tau_{\mathrm{ext}}=1$）**凸二次待决流体外部性** $E_q = \sum_v \Delta_v(q)^2$；
2. **修复四项并发/精度边界缺陷**：
   - 本地背包改用“先精确打包、再二遍均匀发放 `(1 - rho_sys)` 热余量”的两遍机制，彻底消除第一遍加余量挤占同机后续更重任务引起的 Lemma 1 违约；
   - 为容量比较统一注入 IEEE-754 ULP 容差（`cap * (1.0 - 1e-9)` 与 `1e-6 * f_avail[j]`），彻底消除满载边缘浮点误差；
   - 取消正交模式下对边缘时机护盾的不当禁用，使其同步抵御跨拍 CSI 噪声与次拍突发；
   - 将遗传收缩后的 `2M+2` 空间复用填补统一接入完整复合准入谓词 $\Phi(S)$（同时核验 2 拍流地平线与活跃流半松弛盾）。

---

## 三、`CLWB-Dual` 完整最小不可约单核方程与源码 1:1 映射（Step 3.3.1）

### 3.1 四位一体最小耦合方程组
1. **零干扰本地主导二遍精确背包（Local-Dominance Knapsack）**：  
   由于本地计算既不产生同频无线干扰也不消耗边缘算力 $F_j^{\mathrm{avail}}$，记控制生效滞后下本地执行时刻 $t_{\mathrm{exec}}^{\mathrm{loc}} = t_{\mathrm{exec}} + \delta_{\mathrm{sig}}$。对满足 $f_{r,\mathrm{loc}}^{\min} = \max\!\big(f_{\min},\, \frac{C_r}{d_r - t_{\mathrm{exec}}^{\mathrm{loc}}}\big) \le F_{i(r),\mathrm{rem}}^{\mathrm{loc}}(1 - 10^{-9})$ 的任务按 $f_{r,\mathrm{loc}}^{\min}$ 升序精确打包得最大基数集合 $\mathcal{A}_{\mathrm{loc}}$；随后再于同一设备剩余可用余裕 $\max(0, F_{i,\mathrm{rem}}^{\mathrm{loc}} - \sum_{r\in\mathcal{A}_{\mathrm{loc}}^i} f_{r,\mathrm{loc}}^{\min} - 10^{-6}F_i^{\mathrm{loc}})$ 内为 $\mathcal{A}_{\mathrm{loc}}$ 追加不超过 $(1 - \rho_{\mathrm{sys}})f_{r,\mathrm{loc}}^{\min}$ 的频率备份。

2. **半双工空射频因果预去重（Idle-Radio Pacing & Pre-Dedup）**：  
   物理限制每拍每台终端最多激活一条尚未被占用的射频流：在 `pending \ \mathcal{A}_{loc}` 中保留满足 `¬radio_busy_i ∧ q_i^tx = 0` 且在当前残余干预下至少存在一台孤立可行边缘服务器 $j$ 的任务，每台空闲终端至多推举 **1 个最早截止（EDF）孤立可行代表** 进入竞争候选池 $\mathcal{P}_{\mathrm{edge}}(t_k)$，从物理层根除盲目队列积压导致的跨拍失约。

3. **最优信道选机、凸二次双流流体外部性与 Pigouvian-EDF 排序**：  
   对每个候选 $q \in \mathcal{P}_{\mathrm{edge}}(t_k)$ 在其所有孤立可行服务器中选取信道增益最高（平局时所需计算频率最小）的唯一服务器 $j^*(q) = \arg\max_{j \in \mathcal{F}_{\mathrm{iso}}(q)} h_{i(q),j}$。在共享无线子带模式（`orthogonal = False`）下，取每个共信道设备最早截止孤立可行代表构成受害代表集 $\mathcal{P}_{\mathrm{rep}}(t_k)$，精确积分 $q$ 与 $v \in \mathcal{P}_{\mathrm{rep}}(t_k) \setminus \{i(q)\}$ 的两流并行连续流体上传时间 $u_{v \mid q}$：
   $$\Delta_v(q) = \begin{cases} 1, & \text{若 }\mathrm{slack}_{v \mid q} \le 10^{-9}\text{ 或 }\frac{C_v}{\mathrm{slack}_{v \mid q}} > F_{j^*(v),\mathrm{rem}}^{\mathrm{avail}}(1 - 10^{-9}), \\[4pt] \min\!\Big(1,\, \max\!\big(0,\, \frac{f_v^{\min}(q) - f_{v,0}^{\min}}{\max(F_{j^*(v),\mathrm{rem}}^{\mathrm{avail}},\, f_{\min})}\big)\Big), & \text{其他情况} \end{cases}$$
   构造**凸二次跨层流体外部性惩罚**与 **Pigouvian-EDF 有效紧急度**：
   $$E_q = \sum_{v \in \mathcal{P}_{\mathrm{rep}}(t_k)} \Delta_v(q)^2, \qquad k_q^{\mathrm{eff}} = \max(0,\, d_q - t_k)\cdot\big(1 + \max(0,\, E_q - 1)\big)$$
   按 $(k_q^{\mathrm{eff}}, f_{q,j^*}^{\mathrm{iso}}, d_q, \mathrm{id}_q)$ 升序排列形成候选序列，在温和流（$E_q \le 1$）中保持纯 EDF，在 `CE_2ndOrder` / `L3` 等高破坏干扰流（$E_q > 1$）中呈超线性退避让路，而在正交场景下自动退化为 $E_q \equiv 0$。

4. **统一遗传复合谓词 $\Phi(S)$、`O(log P)` 收缩 / `O(M)` 填补与非囤积时机护盾**：  
   在精确连续流体校验器（在线预测滞后 $L_{\mathrm{pred}} = (t_{\mathrm{cur}} - t_{\mathrm{enter}}) + \min(\hat{L}_{\mathrm{verify}}, \delta_{\mathrm{sig}})$）上构建下闭（遗传反单调）复合准入谓词 $\Phi(S)$：
   - 连续流体全局精确可行（所有 $r \in \mathcal{U}(t_k) \cup S$ 同时满足 DDL、NIC 排他、$\rho_{\mathrm{sys}}F_j^{\max}$ 服务器总封顶）；
   - 在非正交模式下，若 $q \in S$ 属于可延后有限算力流（推迟至 $t_k + \Delta t$ 仍孤立可行且 $f_{q,j^*}^{\mathrm{iso}}/F_{j^*}^{\mathrm{avail}} > 10^{-3}$），须满足单流清道地平线 $t_q^{\mathrm{tx,e}} \le t_k + 2\Delta t$；
   - 对非正交活跃在空流 $u \in \mathcal{U}(t_k)$，须保持剩余计算余量不少于半松弛底线 $s_{u,\min} = \min(0.5 s_{u,0},\, \Delta t)$。

   若全列表过关则直接全量接纳；否则对截去末尾高惩罚者的前缀执行二分前缀收缩（至多 $\lceil\log_2(P+1)\rceil$ 次校验），再在同一谓词 $\Phi(\cdot)$ 下对剩余候选执行至多 $2M+2$ 次空间复用填补（允许不与高优先级冲突的异子带/轻量短包搭车进入），并在返回前执行一次实时墙钟刷新校验（必要时再二分微缩）。最终在可用空闲余裕 $\mathrm{headroom}_j = \max(0, F_{j,\mathrm{rem}}^{\mathrm{avail}} - \sum f_r^{\mathrm{base}} - 10^{-6}F_j^{\mathrm{avail}})$ 内，为每个准入流叠加目标覆盖 $s_r^{\mathrm{shield}} = \min(\Delta t + u_r^{\mathrm{est}},\, 0.5\,\mathrm{slack}_r^{\mathrm{cmp}})$ 的非囤积频率护盾（单流增量上限 $\le f_r^{\mathrm{base}}$）。

### 3.2 方程—源码 1:1 映射与 AST 静态审计（Gate 2）
| 算法模块 / 公式 | [champion_core.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage3_algorithm/champion_core.py) 对应符号与行号 | AST 分支数（上限 24） |
| :--- | :--- | :---: |
| 连续流体积分与半双工 NIC 推进 | `_dispatch_device_queues` (`L31–42`), `_init_rollout_state` (`L45–122`), `_compute_run_rates` (`L125–150`), `_run_fluid_rollout` (`L153–218`) | `2, 3, 4, 8` |
| 连续流体可行性与实时算力反演 | `_check_fluid_feasibility` (`L221–266`), `_fast_fluid_verify` (`L269–283`) | `7, 1` |
| 本地主导二遍精确背包（Lemma 1） | `CLWBDualPolicy._admit_local_dominant` (`L302–343`) | `6` |
| 凸二次双流连续流体外部性 $E_q$ | `CLWBDualPolicy._compute_two_flow_externality` (`L345–412`) | `4` |
| 空射频预去重与 Pigouvian-EDF 排序 | `CLWBDualPolicy._pre_dedup_idle_indices` (`L415–432`), `_build_candidate_order` (`L434–530`) | `3, 8` |
| 活跃半松弛盾与复合谓词 $\Phi(S)$ | `CLWBDualPolicy._compute_active_slack_floor` (`L532–553`), `_is_batch_admissible` (`L555–577`) | `4, 6` |
| $O(\log P)$ 遗传收缩与 $O(M)$ 填补 | `CLWBDualPolicy._contract_prefix` (`L579–595`), `_hereditary_select_grants` (`L597–668`) | `2, 10` |
| 非囤积拍+上传时机护盾算力锁定 | `CLWBDualPolicy._allocate_shielded_edge` (`L670–706`), `__call__` (`L708–748`) | `4, 7` |

- **Gate 2 AST 静态审计**（[stage3_results.json](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage3_algorithm/stage3_results.json)）：`bad_magic_numbers = []`（`0` 个未推导魔数），`bad_dataset_strings = []`（`0` 个场景分支词），`max_function_branches = 11`（远低于 `24` 上限，`overcomplex_functions = []`）。

---

## 四、已冻结交由 Stage-4 万次证伪审计的形式化定理清单（Exit Gate 8）

为保证理论陈述与实证条件 100% 严格对齐，正式立项并交由 Stage-4 接受万次随机反例证伪的 5 条性质及显式成立假设如下：

1. **Lemma 1（零干扰本地主导二遍背包最优性）**：
   - **前提**：本地任务共享单终端独立 CPU 容量 $F_i^{\mathrm{loc}}$，不进入无线信道干扰矩阵 $\mathbf{H}$ 且不占用边缘算力 $F_j^{\mathrm{avail}}$，控制生效延迟受限于 $t_{\mathrm{exec}} + \delta_{\mathrm{sig}}$。
   - **结论**：按最小所需本地算力 $f_{r,\mathrm{loc}}^{\min}$ 升序二遍贪心打包（首遍基数最大化、次遍残余释放）严格最大化单机单拍本地可按时交付任务数，同时完整保留稀缺频谱与边缘算力给无法本地完成的任务。
2. **Theorem 1（复合遗传谓词可行性与单拍零违约保护）**：
   - **前提**：(i) 信道状态完美可知（$\hat{\mathbf{H}} = \mathbf{H}$）；(ii) 在线预测控制延迟上界 $L_{\mathrm{pred}}$ 覆盖校验剩余墙钟消耗；(iii) 决策点在空流集合 $\mathcal{U}(t_k)$ 当前孤立静默可行。
   - **结论**：复合谓词 $\Phi(S)$ 在 $\mathcal{P}_{\mathrm{edge}}(t_k)$ 上满足遗传反单调性（阶段二定理 2.1 的直接推论）；输出准入集合 $\mathcal{G}(t_k)$ 与 $\mathcal{U}(t_k) \cup \mathcal{G}(t_k)$ `100%` 满足所有终端-服务器硬资源上限、半双工 NIC 排他性、$\rho_{\mathrm{sys}}F_j^{\max}$ 服务器总封顶及在空活跃链路按时完成约束，零单拍低估违约与零非法动作。
3. **Theorem 2（多级决策计算复杂度与实测零跳拍实时界）**：
   - **前提**：单拍边缘候选规模 $P = |\mathcal{P}_{\mathrm{edge}}(t_k)|$，终端数 $N$，边缘服务器数 $M$，单次连续流体事件数以 $O(N + |\mathcal{U}|)$ 为界。
   - **结论**：向量化双流外部性计算耗时严格为 $O(P^2 + P|\mathcal{U}|)$；单拍对连续流体校验器 `_fast_fluid_verify` 的总调用次数在最坏刷新场景下严格受限于
     $$N_{\mathrm{verify}} \le 2\lceil \log_2(P + 1)\rceil + 2M + 5 = O(\log P + M)$$
     （其中包含基线扫描 $1$ 次、全集检查 $1$ 次、主前缀收缩至多 $\lceil \log_2(P+1)\rceil$ 次、空间填补至多 $2M+2$ 次、终检 $1$ 次及终检失败极小回缩至多 $\lceil \log_2(P+1)\rceil$ 次），彻底根除 `SFA/IBR_5` 的 $O(P^2 M)$ 组合爆炸，并在 $P \in [10, 100]$ 下控制 P99 壁钟时延 $< \Delta t = 50\text{ ms}$ 且跳拍数为 `0`。
4. **Theorem 3（四大退化极限精确还原与核心反例严格支配）**：
   - **前提**：控制面计算滞后 $L_{\mathrm{solver}} = 0$，输入场景退化至 `Model v1.0` 四大极限锚点（Anchor 1：正交无干扰统一截止期单服务器；Anchor 2：无限边缘 CPU 对称载荷统一截止期多流竞争；Anchor 3：无限带宽 $M=1$ 统一截止期计算打包；Anchor 4：单任务全参数），或对应 `L1–L3`、`CE_2ndOrder`。
   - **结论**：在四大退化极限下，`CLWB-Dual` 准入基数 **精确等于** 全知单拍离散最优求解器 `solve_single_tick_exact`（若截止期非统一且倒置，在 Anchor 1 下与最小份额最优背包间存在可控经验间隙；若运行在 $L_{\mathrm{solver}} > 0$ 真实壁钟边界，容量比 $> 0.998$ 处亦存在微小时间留白差异）；同时在 `L2/L3/CE_2ndOrder` 上严格追平 Oracle (`3/3`, `3/4`, `4/5`)。
5. **Proposition 4（非囤积护盾抗 CSI 噪声与跨拍突发机制）**：
   - **前提**：上行存在对数正态信道测量噪声（如 $\sigma_{\mathrm{dB}} = 2.0\text{ dB}$）及次拍突发并发切入。
   - **结论**：活跃链路半松弛底线与不超 $f_r^{\mathrm{base}}$ 封顶的非囤积时机护盾在保留后续余量的前提下，显著压缩两拍并发叠加 CSI 波动引发的边缘失约率（在 `WK11_CSI_NOISE_2DB` 中实现 `96.06%` 全周期按时率与 `2.85%` 失约率）。

---

## 五、全量五道门禁实证结果总表（数据源：[stage3_results.json](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage3_algorithm/stage3_results.json)）

### 5.1 Gate 1：健全性、Oracle 对拍、退化锚点与突发壁钟缩放
1. **反例免疫核验（`CE: 4/4 passed`）**：
   - `L1`：`CLWB_Dual = 2/4`（Oracle `3/4`，`DGOLB+CG = 0/4`，`EDF_PreBatch+CG = 0/4`）；
   - `L2`：`CLWB_Dual = 3/3 == Oracle`（非延迟理论上限 `2/3`）；
   - `L3`：`CLWB_Dual = 3/4 == Oracle`（`SFA+CG = 1/4`）；
   - `CE_2ndOrder`：**`CLWB_Dual = 4/5 == Oracle`**（`SFA+CG = 2/5`，`DGOLB+CG = 1/5`，`Chen_Game+CG = 2/5`）。
2. **100 组随机微算例精确 Oracle 对拍**：全知 Oracle 平均 `100.00%`，`CLWB_Dual` 捕获率 **`98.75%`**（`>= 95.00%`），基线反超例数 `inversions = 0`，非法动作 `0`，累计通过 `2,996` 次硬件硬约束运行时断言。
3. **四大退化极限锚点**：Anchor 1 (`2/2`)、Anchor 2 (`3/3`)、Anchor 3 (`2/2`)、Anchor 4 (`1/1`) **`4/4` 精确匹配**。
4. **突发规模 $P \in \{10, 25, 45, 70, 100\}$ 实时壁钟延迟测试**：

| 突发并发规模 $P$ | 按时完成率 (`ontime_rate`) | 边缘准入失约率 (`miss_rate`) | 单步决策 P99 壁钟耗时 (`decision_p99_ms`) | 控制面跳拍 (`skipped_ticks`) | 硬断言核验次数 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `10` | `100.00%` | `0.00%` | `0.95 ms` | `0` | `70` |
| `25` | `100.00%` | `0.00%` | `1.29 ms` | `0` | `140` |
| `45` | `100.00%` | `0.00%` | `9.33 ms` | `0` | `376` |
| `70` | `97.14%` | `0.00%` | `17.67 ms` | `0` | `659` |
| `100` | `99.00%` | `0.00%` | **`19.37 ms`**（$\ll 50\text{ ms}$） | **`0`** | `904` |

### 5.2 Gate 3：主基准 $K=12$ 独立工况全双轨 `+CommonGuard` 对决与 Holm-Wilcoxon 校正检验
> **统计功效锁死核验**：独立工况 $K=12$（每工况内以 3 随机种子均值作为单一独立样本点），对比基线数 $m=7$，理论最小可达家族校正 $p$ 值为 $p_{\min}^{\mathrm{Holm}} = 7 \times 2^{-11} = 0.00342 \le 0.02$。

| 算法（外部对手均公平挂载 `+CG`） | $K=12$ 均值按时率 | 边缘失约率 | 服务器废算率 | 最大 P99 耗时 | 累计跳拍 | 相对净增益 ($\Delta$ pp) | 胜场 ($/12$) | 原始 Wilcoxon $p$ | **Holm 校正 $p_{\mathrm{Holm}}$** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CLWB_Dual` (本篇最小单核)** | **`98.01%`** | **`0.36%`** | **`0.34%`** | `21.16 ms` | **`0`** | — | — | — | — |
| `Bambos_ALP+CG` *(经典无线 ALP 2000)* | `89.08%` | `0.00%` | `0.00%` | `12.21 ms` | `0` | **`+8.93 pp`** | **`12/12`** | `0.00024` | **`0.00171 < 0.05`** |
| `Chen_Game+CG` *(经典势博弈 2016)* | `66.72%` | `26.37%` | `23.14%` | `17.49 ms` | `0` | **`+31.29 pp`** | **`12/12`** | `0.00024` | **`0.00171 < 0.05`** |
| `EDF_PreBatch+CG` *(经典 EDF 1973)* | `34.86%` | `89.17%` | `82.32%` | `2.77 ms` | `0` | **`+63.15 pp`** | **`12/12`** | `0.00024` | **`0.00171 < 0.05`** |
| `LLF_PreBatch+CG` *(经典 LLF 1983)* | `34.72%` | `89.37%` | `82.60%` | `3.48 ms` | `0` | **`+63.29 pp`** | **`12/12`** | `0.00024` | **`0.00171 < 0.05`** |
| `DGOLB+CG` *(Stage-1 SOTA, TMC 2026)* | `23.21%` | `91.45%` | `86.12%` | `5.50 ms` | `0` | **`+74.80 pp`** | **`12/12`** | `0.00024` | **`0.00171 < 0.05`** |
| `SFA+CG` *(序贯精确准入)* | `11.79%` | `86.74%` | `67.08%` | `9187.07 ms` | `996` | **`+86.22 pp`** | **`12/12`** | `0.00024` | **`0.00171 < 0.05`** |
| `IBR_5+CG` *(5 轮迭代重优化)* | `2.11%` | `96.55%` | `83.66%` | `12282.42 ms` | `2172` | **`+95.90 pp`** | **`12/12`** | `0.00024` | **`0.00171 < 0.05`** |

#### $K=12$ 独立工况逐项按时完成率明细表（3 种子均值）

| 独立工况 (`wid`) | `CLWB_Dual` | `Bambos_ALP+CG` | `Chen_Game+CG` | `EDF_PreBatch+CG` | `DGOLB+CG` | `SFA+CG` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `WK01_BALANCED_MED` | **`100.00%`** | `99.07%` | `79.86%` | `38.43%` | `31.71%` | `20.83%` |
| `WK02_DENSE_TIGHT` | **`96.00%`** | `77.78%` | `63.78%` | `28.89%` | `18.67%` | `12.89%` |
| `WK03_DEADLINE_TIGHT` | **`93.29%`** | `78.01%` | `52.08%` | `24.54%` | `15.97%` | `13.66%` |
| `WK04_HETERO_DEV` | **`98.38%`** | `89.58%` | `60.42%` | `31.48%` | `23.61%` | `11.81%` |
| `WK05_HETERO_DL` | **`99.76%`** | `94.76%` | `70.00%` | `34.29%` | `26.19%` | `20.00%` |
| `WK06_BURST_MODERATE` | **`99.58%`** | `87.08%` | `64.38%` | `34.58%` | `21.25%` | `3.54%` |
| `WK07_BURST_HEAVY` | **`98.79%`** | `83.43%` | `61.41%` | `33.13%` | `16.16%` | `2.22%` |
| `WK08_HIGH_PAYLOAD` | **`97.22%`** | `89.17%` | `73.06%` | `38.33%` | `26.39%` | `17.22%` |
| `WK09_SMALL_CELL_ASYM` | **`98.70%`** | `82.55%` | `64.58%` | `33.07%` | `21.88%` | `21.35%` |
| `WK10_DUAL_SUBBAND` | **`99.26%`** | `97.78%` | `65.93%` | `41.11%` | `28.52%` | `6.67%` |
| `WK11_CSI_NOISE_2DB` | **`96.06%`** | `93.98%` | `65.51%` | `35.65%` | `27.08%` | `11.34%` |
| `WK12_LARGE_SCALE` | **`99.07%`** | `95.74%` | `79.63%` | `44.81%` | `21.11%` | `0.00%` |

### 5.3 Gate 4：留出分布外（`K_OOD = 4`）泛化与最差子群保护

| 分布外测试集 | `CLWB_Dual` | `Bambos_ALP+CG` | `Chen_Game+CG` | `EDF_PreBatch+CG` | `LLF_PreBatch+CG` | `DGOLB+CG` | `SFA+CG` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `OOD1_ASYM_NEARFAR` | **`93.33%`** | `79.02%` | `65.29%` | `31.57%` | `31.57%` | `23.14%` | `15.29%` |
| `OOD2_HETDL_CSI2DB` | **`98.10%`** | `96.19%` | `57.86%` | `30.48%` | `30.24%` | `21.19%` | `20.48%` |
| `OOD3_EXTREME_BURST75` | **`98.67%`** | `89.78%` | `78.89%` | `39.56%` | `39.33%` | `14.00%` | `0.00%` |
| `OOD4_SPARSE_SRV` | **`93.33%`** | `77.95%` | `68.72%` | `29.74%` | `29.49%` | `21.28%` | `26.15%` |
| **OOD 均值 (`mean`)** | **`95.86%`** | `85.73%` | `67.69%` | `32.84%` | `32.66%` | `19.90%` | `15.48%` |
| **最差子群 (`worst`)** | **`93.33%`** | `77.95%` | `57.86%` | `29.74%` | `29.49%` | `14.00%` | `0.00%` |

### 5.4 Gate 5：三向 Leave-One-Out (LOO) 零负优化消融验证

| 消融变体 | $K=12$ 均值按时率 | 完整体净贡献 ($\Delta$ pp) | 失约率 (`miss`) | `L1` | `L2` | `L3` | `CE_2ndOrder` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CLWB_Dual` (完整最小单核)** | **`98.01%`** | — | **`0.36%`** | **`2/4`** | **`3/3`** | **`3/4`** | **`4/5`** |
| `LOO_no_externality_pricing` *(撤去凸二次外部性)* | `97.92%` | **`+0.09 pp`** | `0.41%` | `2/4` | `3/3` | **`1/4` (崩)** | **`2/5` (崩)** |
| `LOO_no_lagrangian_defer` *(撤去因果空射频与2拍/半松弛)* | `69.47%` | **`+28.54 pp`** | `6.65%` | `2/4` | **`1/3` (崩)** | `3/4` | **`1/5` (崩)** |
| `LOO_no_fluid_contraction` *(撤去连续流体契约与护盾)* | `38.22%` | **`+59.79 pp`** | `84.66%` | **`0/4` (崩)** | **`0/3` (崩)** | **`0/4` (崩)** | **`0/5` (崩)** |

---

## 六、阶段三准出 8 项硬门禁逐项签收表

- [x] **1. 底座忠实度与二阶反例防浅核验**：`8/8` Stage-2 SHA-256 哈希零改动；全部常识修补隔离至 [common_guard.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage3_algorithm/common_guard.py)；精确追平二阶反例 `CE_2ndOrder = 4/5 == Oracle`。
- [x] **2. 数学杠杆驱动与精简耦合**：从三路线对比出发并经历强敌与反拼盘双轮锤炼，最终收敛至最小不可约四位一体核 `CLWB-Dual`。
- [x] **3. 三向 LOO 全正且各锁核心反例**：三项主要杠杆 LOO 贡献分别为 `+0.09 pp / +28.54 pp / +59.79 pp`，各自唯一守住 `L3`/`CE_2ndOrder`/`L2`/`L1`。
- [x] **4. 单核零魔数 AST 审计**：未推导魔数为 `0`、场景分支词为 `0`、最大函数分支数为 `11 <= 24`。
- [x] **5. 健全性与真机预算 100% 通过**：[soundness_test.py](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage3_algorithm/soundness_test.py) 全 pass，`micro_100` Oracle 捕获 `98.75%`（`0` 反超、`0` 非法、`2,996` 硬断言零违约），`P=100` P99 耗时 `19.37 ms`，`skipped_ticks = 0`。
- [x] **6. `CommonGuard` 边界严明**：主表 7 个对照基线统一挂载 `+CG`，无任何 `OurCore` 算子泄漏。
- [x] **7. 统计功效预设达标**：独立样本 $K=12$ 相对全部 7 强敌取得 `12/12` 全胜，$p_{\mathrm{Holm}} = 0.00171 < 0.05$；$K_{\mathrm{OOD}}=4$ 均值与最差子群（`95.86% / 93.33%`）大幅领先最优基线（`85.73% / 77.95%`）。
- [x] **8. 待证伪定理清单与 Stage-4 全绿认证闭环**：已交付 Stage-4 并通过 5 张红灯表全绿审计（详见 [STAGE4_AUDIT_REPORT.md](file:///c:/Users/17336/Desktop/my_study/论文资料/我的论文/论文4/重构_v3/stage4_audit/STAGE4_AUDIT_REPORT.md)）。
