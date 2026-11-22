"""Formal Read-Only System & Physics Engine for Edge Uplink-Compute Batch Scheduling (Model v1.0).

Pure mathematical/physical problem specification engine:
  * Strictly zero new algorithm logic or heuristic rules.
  * Continuous fluid-SINR uplink integration (event-driven reflow on every link join/exit/timeout)
    with machine-precision bit conservation (|delivered - D| / D <= 1e-8).
  * Strictly causal observation mask `observe(t_k)` (zero future completion `end_s` leaks).
  * Real control-plane actuation lag `t_exec = t_k + L(t_k) + delta_sig` with single-threaded
    controller tick-skipping when `L(t_k) + delta_sig > tick_s`.
  * Hard physical runtime assertions checked after every simulation event (NIC half-duplex,
    server/device CPU caps with rho_sys=0.95, causality order, non-negative remaining bits).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import time
from typing import Callable, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Immutable physical constants and data structures
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LinkPhysics:
    bandwidth_hz: float = 10_000_000.0
    noise_w: float = 1e-13
    g0: float = 0.1
    pathloss_exp: float = 4.0
    p_ms_w: float = 1.0
    alpha_dev: float = 1e-27
    alpha_ms: float = 1e-27


@dataclass(frozen=True)
class TaskSpec:
    tid: str
    device: int
    k: int
    arrival_s: float
    deadline_s: float
    D_bits: float
    B_out_bits: float
    C_cycles: float
    tau_dn_s: float


@dataclass(frozen=True)
class SimConfig:
    tick_s: float = 0.05
    delta_sig_s: float = 0.001
    rho_sys: float = 0.95
    f_min_hz: float = 1e6
    latency_mode: str = "wall"   # "wall" | "fixed" | "zero"
    fixed_latency_s: float = 0.0
    n_subbands: int = 1
    orthogonal: bool = False
    csi_sigma_db: float = 0.0
    csi_seed: int = 0
    wall_clock_limit_s: float = 30.0
    horizon_extra_s: float = 0.2


@dataclass
class ScenarioV1:
    case_id: str
    family: str
    seed: int
    n_devices: int
    n_servers: int
    tasks: list[TaskSpec]
    ms_pos: np.ndarray            # (M, 2)
    dev_pos: np.ndarray           # (N, 2)
    home_ms: np.ndarray           # (N,)
    d_ms: np.ndarray              # (N, M)
    h_dev_ms: np.ndarray          # (N, M)
    f_loc_hz: np.ndarray          # (N,)
    f_ms_max_hz: np.ndarray       # (M,)
    p_tx_w: np.ndarray            # (N,)
    subband: np.ndarray           # (N,)
    phys: LinkPhysics
    T_max_s: float
    window_s: float

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)


# --------------------------------------------------------------------------- #
# Physical primitives and scenario constructors
# --------------------------------------------------------------------------- #

def shannon_rate(bw_hz: float, signal_w: float, noise_w: float, interf_w: float) -> float:
    """Instantaneous Shannon capacity B * log2(1 + S / (N0 + I)), bounded > 0."""
    denom = max(float(noise_w) + max(float(interf_w), 0.0), 1e-30)
    sinr = max(float(signal_w), 0.0) / denom
    return max(float(bw_hz) * math.log2(1.0 + sinr), 1e-3)


def build_scenario_v1(
    *,
    case_id: str = "C01",
    family: str = "balanced",
    seed: int = 1001,
    n_devices: int = 24,
    n_servers: int = 6,
    tasks_per_device: int = 6,
    offered_load: float = 0.60,
    T_max_s: float = 1.0,
    radius_m: float = 300.0,
    area_m: float = 5000.0,
    f_dev_range_ghz: tuple[float, float] = (0.70, 0.90),
    f_ms_ghz: float = 50.0,
    bandwidth_hz: float = 10_000_000.0,
    noise_w: float = 1e-13,
    g0: float = 0.1,
    pathloss_exp: float = 4.0,
    p_tx_range_w: tuple[float, float] = (0.8, 1.3),
    cycles_range: tuple[float, float] = (1e8, 1e9),
    input_kb_range: tuple[float, float] = (60.0, 120.0),
    output_kb_range: tuple[float, float] = (0.1, 10.0),
    hetero_deadline_range: Optional[tuple[float, float]] = None,
    n_subbands: int = 1,
    window_s: Optional[float] = None,
    rho_sys: float = 0.95,
    interf_scale: float = 1.0,
) -> ScenarioV1:
    """Deterministic exogenous scenario calibrated to Stage-1 wireless edge physics."""
    rng = np.random.default_rng(int(seed))
    n = int(n_devices)
    m = int(n_servers)
    q = int(tasks_per_device)

    ms_pos = rng.uniform(0.0, float(area_m), size=(m, 2))
    home = rng.integers(0, m, size=n)
    rr = float(radius_m) * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    th = rng.uniform(0.0, 2.0 * np.pi, size=n)
    dev_pos = ms_pos[home] + np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)

    d_ms = np.linalg.norm(dev_pos[:, None, :] - ms_pos[None, :, :], axis=2)
    # 3GPP minimum UE-BS horizontal coupling separation (>= 10 m) avoids 1 m singular spikes
    h_dev_ms = float(g0) * (1.0 / np.maximum(d_ms, 10.0)) ** float(pathloss_exp)
    if interf_scale != 1.0:
        # Preserves intended signal scale by scaling path-loss gains via effective spacing
        # only if caller requested custom topology density; keep raw h_dev_ms unmodified
        # when interf_scale == 1.0.
        h_dev_ms = h_dev_ms * float(interf_scale)

    f_loc_hz = rng.uniform(float(f_dev_range_ghz[0]), float(f_dev_range_ghz[1]), size=n) * 1e9
    f_ms_max_hz = np.full(m, float(f_ms_ghz) * 1e9, dtype=float)
    p_tx_w = rng.uniform(float(p_tx_range_w[0]), float(p_tx_range_w[1]), size=n)
    subband = np.arange(n, dtype=int) % max(1, int(n_subbands))

    C_mat = rng.uniform(float(cycles_range[0]), float(cycles_range[1]), size=(n, q))
    D_mat = rng.uniform(float(input_kb_range[0]), float(input_kb_range[1]), size=(n, q)) * 8.0 * 1e3
    B_mat = rng.uniform(float(output_kb_range[0]), float(output_kb_range[1]), size=(n, q)) * 8.0 * 1e3

    if window_s is None:
        mean_c = 0.5 * (float(cycles_range[0]) + float(cycles_range[1]))
        sys_cap = float(np.sum(f_loc_hz) + float(rho_sys) * np.sum(f_ms_max_hz))
        window_s = max(0.05, (n * q * mean_c) / (max(float(offered_load), 1e-4) * sys_cap))

    arr_mat = rng.uniform(0.0, float(window_s), size=(n, q))
    arr_mat.sort(axis=1)

    if hetero_deadline_range is not None:
        lo, hi = float(hetero_deadline_range[0]), float(hetero_deadline_range[1])
        u = rng.uniform(lo, hi, size=(n, q))
        u = u / max(float(np.mean(u)), 1e-9)
        dl_mat = arr_mat + float(T_max_s) * np.maximum(u, 0.1)
    else:
        dl_mat = arr_mat + float(T_max_s)

    phys = LinkPhysics(
        bandwidth_hz=float(bandwidth_hz),
        noise_w=float(noise_w),
        g0=float(g0),
        pathloss_exp=float(pathloss_exp),
        p_ms_w=1.0,
        alpha_dev=1e-27,
        alpha_ms=1e-27,
    )

    tasks: list[TaskSpec] = []
    for i in range(n):
        h_home = float(h_dev_ms[i, int(home[i])])
        r_dn = shannon_rate(phys.bandwidth_hz, phys.p_ms_w * h_home, phys.noise_w, 0.0)
        for k in range(q):
            b_bits = float(B_mat[i, k])
            tau_dn = b_bits / r_dn
            tasks.append(
                TaskSpec(
                    tid=f"{i}:{k}",
                    device=int(i),
                    k=int(k),
                    arrival_s=float(arr_mat[i, k]),
                    deadline_s=float(dl_mat[i, k]),
                    D_bits=float(D_mat[i, k]),
                    B_out_bits=b_bits,
                    C_cycles=float(C_mat[i, k]),
                    tau_dn_s=float(tau_dn),
                )
            )
    tasks.sort(key=lambda t: (t.arrival_s, t.deadline_s, t.device, t.k))

    h_dev_ms.setflags(write=False)
    ms_pos.setflags(write=False)
    dev_pos.setflags(write=False)
    d_ms.setflags(write=False)

    return ScenarioV1(
        case_id=str(case_id),
        family=str(family),
        seed=int(seed),
        n_devices=n,
        n_servers=m,
        tasks=tasks,
        ms_pos=ms_pos,
        dev_pos=dev_pos,
        home_ms=home,
        d_ms=d_ms,
        h_dev_ms=h_dev_ms,
        f_loc_hz=f_loc_hz,
        f_ms_max_hz=f_ms_max_hz,
        p_tx_w=p_tx_w,
        subband=subband,
        phys=phys,
        T_max_s=float(T_max_s),
        window_s=float(window_s),
    )


def make_micro_scenario(
    *,
    case_id: str,
    n_devices: int,
    n_servers: int,
    h_dev_ms: np.ndarray,
    f_loc_hz: np.ndarray,
    f_ms_max_hz: np.ndarray,
    p_tx_w: np.ndarray,
    tasks_raw: list[dict],
    bandwidth_hz: float = 10_000_000.0,
    noise_w: float = 1e-13,
    n_subbands: int = 1,
    seed: int = 0,
    family: str = "micro",
) -> ScenarioV1:
    """Explicit deterministic micro-scenario constructor for exact oracles & counterexamples."""
    n = int(n_devices)
    m = int(n_servers)
    h_arr = np.array(h_dev_ms, dtype=float, copy=True)
    assert h_arr.shape == (n, m)
    f_loc = np.array(f_loc_hz, dtype=float, copy=True)
    f_ms = np.array(f_ms_max_hz, dtype=float, copy=True)
    ptx = np.array(p_tx_w, dtype=float, copy=True)
    home = np.argmax(h_arr, axis=1)
    subband = np.arange(n, dtype=int) % max(1, int(n_subbands))
    phys = LinkPhysics(bandwidth_hz=float(bandwidth_hz), noise_w=float(noise_w))

    tasks: list[TaskSpec] = []
    max_span = 0.0
    max_arr = 0.0
    for idx, tr in enumerate(tasks_raw):
        i = int(tr["device"])
        k = int(tr.get("k", idx))
        arr = float(tr["arrival_s"])
        dl = float(tr["deadline_s"])
        d_bits = float(tr["D_bits"])
        c_cyc = float(tr["C_cycles"])
        b_bits = float(tr.get("B_out_bits", 800.0))
        tau_dn = float(tr.get("tau_dn_s", 0.001))
        max_span = max(max_span, dl - arr)
        max_arr = max(max_arr, arr)
        tasks.append(
            TaskSpec(
                tid=str(tr.get("tid", f"{i}:{k}")),
                device=i,
                k=k,
                arrival_s=arr,
                deadline_s=dl,
                D_bits=d_bits,
                B_out_bits=b_bits,
                C_cycles=c_cyc,
                tau_dn_s=tau_dn,
            )
        )
    tasks.sort(key=lambda t: (t.arrival_s, t.deadline_s, t.device, t.k))

    ms_pos = np.zeros((m, 2), dtype=float)
    dev_pos = np.zeros((n, 2), dtype=float)
    d_ms = (phys.g0 / np.maximum(h_arr, 1e-30)) ** (1.0 / phys.pathloss_exp)
    h_arr.setflags(write=False)

    return ScenarioV1(
        case_id=str(case_id),
        family=str(family),
        seed=int(seed),
        n_devices=n,
        n_servers=m,
        tasks=tasks,
        ms_pos=ms_pos,
        dev_pos=dev_pos,
        home_ms=home,
        d_ms=d_ms,
        h_dev_ms=h_arr,
        f_loc_hz=f_loc,
        f_ms_max_hz=f_ms,
        p_tx_w=ptx,
        subband=subband,
        phys=phys,
        T_max_s=max(max_span, 0.05),
        window_s=max(max_arr, 0.05),
    )


def build_calibrated_suite(seed: int = 2026) -> list[tuple[ScenarioV1, SimConfig]]:
    """Standard calibrated scenario suite grounded in frozen Stage-1 wireless-edge physics."""
    return [
        (
            build_scenario_v1(
                case_id="CAL_BALANCED",
                family="balanced",
                seed=seed + 1,
                n_devices=24,
                n_servers=6,
                tasks_per_device=6,
                offered_load=0.60,
                T_max_s=1.0,
                radius_m=300.0,
                f_dev_range_ghz=(0.70, 0.90),
            ),
            SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall"),
        ),
        (
            build_scenario_v1(
                case_id="CAL_INTERF_DENSE",
                family="interference_dense",
                seed=seed + 2,
                n_devices=30,
                n_servers=4,
                tasks_per_device=5,
                offered_load=0.75,
                T_max_s=0.60,
                radius_m=150.0,
                f_dev_range_ghz=(0.70, 0.90),
            ),
            SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall"),
        ),
        (
            build_scenario_v1(
                case_id="CAL_DEADLINE_TIGHT",
                family="deadline_tight",
                seed=seed + 3,
                n_devices=24,
                n_servers=6,
                tasks_per_device=6,
                offered_load=0.70,
                T_max_s=0.60,
                radius_m=220.0,
                f_dev_range_ghz=(0.70, 0.90),
            ),
            SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall"),
        ),
        (
            build_scenario_v1(
                case_id="CAL_HETERO_DEV",
                family="hetero_device",
                seed=seed + 4,
                n_devices=24,
                n_servers=6,
                tasks_per_device=6,
                offered_load=0.70,
                T_max_s=0.80,
                radius_m=250.0,
                f_dev_range_ghz=(0.30, 1.30),
            ),
            SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall"),
        ),
        (
            build_scenario_v1(
                case_id="CAL_BURST_HEAVY",
                family="burst_heavy",
                seed=seed + 5,
                n_devices=50,
                n_servers=6,
                tasks_per_device=3,
                offered_load=0.90,
                T_max_s=0.70,
                radius_m=220.0,
                f_dev_range_ghz=(0.60, 0.90),
            ),
            SimConfig(tick_s=0.05, delta_sig_s=0.001, latency_mode="wall"),
        ),
    ]


def compute_dimensionless_numbers(scen: ScenarioV1, cfg: SimConfig) -> dict[str, float]:
    """Extract the 3 normalized dimensionless governing numbers (kappa_ul, rho_cmp, gamma_tick)."""
    n = scen.n_devices
    m = scen.n_servers
    phys = scen.phys
    sb = scen.subband

    # Effective concurrent batch devices per subband under a control-cycle burst
    n_sb = max(1, int(cfg.n_subbands))
    devs_per_sb = max(1.0, float(n) / float(n_sb))
    k_conc = max(1, int(math.floor(0.5 * devs_per_sb))) if not cfg.orthogonal else 0

    u0_list = []
    uint_list = []
    slack_list = []

    best_j = np.argmax(scen.h_dev_ms, axis=1)
    f_avail_arr = float(cfg.rho_sys) * scen.f_ms_max_hz

    for t in scen.tasks:
        i = t.device
        j = int(best_j[i])
        sig = float(scen.p_tx_w[i] * scen.h_dev_ms[i, j])
        r0 = shannon_rate(phys.bandwidth_hz, sig, phys.noise_w, 0.0)
        u0 = t.D_bits / r0
        u0_list.append(u0)

        if cfg.orthogonal or k_conc <= 0:
            interf = 0.0
        else:
            # Expected co-subband interference from k_conc concurrent batch transmitters at server j
            peers = [
                float(scen.p_tx_w[u] * scen.h_dev_ms[u, j])
                for u in range(n)
                if u != i and (int(sb[u]) % n_sb) == (int(sb[i]) % n_sb)
            ]
            interf = float(k_conc * np.mean(peers)) if peers else 0.0

        rint = shannon_rate(phys.bandwidth_hz, sig, phys.noise_w, interf)
        uint = t.D_bits / rint
        uint_list.append(uint)

        t_cmp_min = t.C_cycles / max(float(f_avail_arr[j]), 1e-9)
        raw_slack = max((t.deadline_s - t.arrival_s) - cfg.delta_sig_s - t.tau_dn_s - t_cmp_min, 1e-3)
        slack_list.append(raw_slack)

    u0_mean = float(np.median(u0_list)) if u0_list else 0.0
    uint_mean = float(np.median(uint_list)) if uint_list else 0.0
    slack_mean = max(float(np.median(slack_list)) if slack_list else 1.0, 1e-6)
    du_batch = max(0.0, uint_mean - u0_mean)

    kappa_ul = du_batch / slack_mean

    total_cycles = float(sum(t.C_cycles for t in scen.tasks))
    sys_cap = float(np.sum(scen.f_loc_hz) + np.sum(f_avail_arr))
    rho_cmp = total_cycles / max(float(scen.window_s) * sys_cap, 1e-9)

    mean_span = float(np.mean([t.deadline_s - t.arrival_s for t in scen.tasks])) if scen.tasks else scen.T_max_s
    gamma_tick = float(cfg.tick_s) / max(mean_span, 1e-9)

    return {
        "kappa_ul": float(kappa_ul),
        "rho_cmp": float(rho_cmp),
        "gamma_tick": float(gamma_tick),
        "u0_s": u0_mean,
        "u_int_s": uint_mean,
        "delta_u_batch_s": du_batch,
        "slack0_mean_s": slack_mean,
        "k_conc_ref": float(k_conc),
    }


# --------------------------------------------------------------------------- #
# Fluid continuous-SINR event-driven simulator with hard-boundary assertions
# --------------------------------------------------------------------------- #

_PRIO_FINISH = 0
_PRIO_DEADLINE = 1
_PRIO_ACTUATE = 2
_PRIO_TICK = 3
_PRIO_HORIZON = 4


class _RuntimeTask:
    __slots__ = (
        "tid", "device", "k", "arrival_s", "deadline_s",
        "D_bits", "B_out_bits", "C_cycles", "tau_dn_s",
        "state", "mode", "server", "f_hz",
        "rem_bits", "integrated_bits", "cur_rate_bps", "tx_ver",
        "t_exec_s", "t_tx_start_s", "t_tx_end_s",
        "t_cmp_start_s", "t_cmp_end_s", "t_done_s",
        "alpha_exec", "energy_cmp_j", "delivered_ok",
    )

    def __init__(self, spec: TaskSpec):
        self.tid = spec.tid
        self.device = spec.device
        self.k = spec.k
        self.arrival_s = spec.arrival_s
        self.deadline_s = spec.deadline_s
        self.D_bits = spec.D_bits
        self.B_out_bits = spec.B_out_bits
        self.C_cycles = spec.C_cycles
        self.tau_dn_s = spec.tau_dn_s

        self.state = "unrevealed"
        self.mode: Optional[str] = None
        self.server: Optional[int] = None
        self.f_hz: Optional[float] = None

        self.rem_bits = spec.D_bits
        self.integrated_bits = 0.0
        self.cur_rate_bps = 0.0
        self.tx_ver = 0

        self.t_exec_s: Optional[float] = None
        self.t_tx_start_s: Optional[float] = None
        self.t_tx_end_s: Optional[float] = None
        self.t_cmp_start_s: Optional[float] = None
        self.t_cmp_end_s: Optional[float] = None
        self.t_done_s: Optional[float] = None

        self.alpha_exec = 0.0
        self.energy_cmp_j = 0.0
        self.delivered_ok = False


class EdgeUplinkEnv:
    """Strict causal continuous-fluid-SINR edge computing simulator."""

    def __init__(self, scen: ScenarioV1, cfg: Optional[SimConfig] = None):
        self.scen = scen
        self.cfg = cfg if cfg is not None else SimConfig()

        self.n = scen.n_devices
        self.m = scen.n_servers
        self.f_avail_hz = float(self.cfg.rho_sys) * scen.f_ms_max_hz

        if self.cfg.csi_sigma_db > 0.0:
            rng_csi = np.random.default_rng(int(self.cfg.csi_seed))
            noise_db = self.cfg.csi_sigma_db * rng_csi.standard_normal(scen.h_dev_ms.shape)
            self._h_obs = scen.h_dev_ms * (10.0 ** (noise_db / 10.0))
        else:
            self._h_obs = np.array(scen.h_dev_ms, dtype=float, copy=True)
        self._h_obs.setflags(write=False)

        self.tasks: list[_RuntimeTask] = [_RuntimeTask(ts) for ts in scen.tasks]
        self.by_id: dict[str, _RuntimeTask] = {t.tid: t for t in self.tasks}

        self.now = 0.0
        self._last_reflow_t = 0.0
        self._seq = 0
        self._heap: list[tuple[float, int, int, str, object]] = []
        self._stop = False

        max_dl = max((t.deadline_s for t in self.tasks), default=1.0)
        self.horizon = float(max_dl + self.cfg.horizon_extra_s)

        # Physical resource state
        self.busy_radios: set[int] = set()
        self.active_uplinks: list[_RuntimeTask] = []
        self.tx_queue: list[list[_RuntimeTask]] = [[] for _ in range(self.n)]
        self.wait_local_q: list[list[_RuntimeTask]] = [[] for _ in range(self.n)]
        self.wait_compute_q: list[list[_RuntimeTask]] = [[] for _ in range(self.m)]
        self.dev_cpu_used = np.zeros(self.n, dtype=float)
        self.srv_cpu_used = np.zeros(self.m, dtype=float)

        # Accounting & invariants
        self.decision_latencies_ms: list[float] = []
        self.skipped_ticks = 0
        self.policy_calls = 0
        self.illegal_actions = 0
        self.n_ontime = 0
        self.n_timeout = 0
        self.n_dropped = 0
        self.n_edge_granted = 0
        self.n_edge_missed = 0
        self.wasted_srv_cycles = 0.0
        self.total_srv_cycles = 0.0
        self.wasted_loc_cycles = 0.0
        self.total_loc_cycles = 0.0
        self.energy_tx_j = 0.0
        self.energy_cmp_j = 0.0
        self.max_bit_rel_err = 0.0
        self.hard_invariant_checks = 0
        self.reflow_count = 0
        self.status = "ok"
        self.error_msg: Optional[str] = None

        for t in self.tasks:
            self._push(t.deadline_s, _PRIO_DEADLINE, "deadline", t.tid)
        self._push(0.0, _PRIO_TICK, "tick", 0)
        self._push(self.horizon, _PRIO_HORIZON, "horizon", None)

    def _push(self, when: float, prio: int, kind: str, payload: object) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (float(when), int(prio), self._seq, kind, payload))

    def _reveal_arrivals(self, now: float) -> None:
        for t in self.tasks:
            if t.state == "unrevealed" and t.arrival_s <= now + 1e-12:
                t.state = "pending"

    def observe(self, now: float) -> dict:
        """Strictly causal controller observation mask F_{t_k} (no future completion times)."""
        srv_reserved = np.zeros(self.m, dtype=float)
        dev_reserved = np.zeros(self.n, dtype=float)
        srv_wait_cnt = np.zeros(self.m, dtype=int)
        tx_queue_bits = np.zeros(self.n, dtype=float)

        need_res = (
            "actuating_edge", "actuating_local",
            "tx_queue", "tx_up", "wait_local", "wait_compute", "compute",
        )
        for t in self.tasks:
            if t.state not in need_res or t.f_hz is None:
                continue
            if t.mode == "edge" and t.server is not None:
                srv_reserved[t.server] += t.f_hz
                if t.state in ("actuating_edge", "tx_queue", "tx_up", "wait_compute"):
                    srv_wait_cnt[t.server] += 1
                if t.state == "tx_queue":
                    tx_queue_bits[t.device] += t.D_bits
            elif t.mode == "local":
                dev_reserved[t.device] += t.f_hz

        pending_obs = [
            {
                "id": t.tid,
                "device": int(t.device),
                "k": int(t.k),
                "arrival_s": float(t.arrival_s),
                "deadline_s": float(t.deadline_s),
                "D_bits": float(t.D_bits),
                "B_out_bits": float(t.B_out_bits),
                "C_cycles": float(t.C_cycles),
                "tau_dn_s": float(t.tau_dn_s),
            }
            for t in self.tasks
            if t.state == "pending"
        ]

        # Note: strictly NO `end_s` field on active_uplinks (zero future-knowledge leak).
        active_obs = [
            {
                "id": t.tid,
                "device": int(t.device),
                "server": int(t.server),
                "p_tx_w": float(self.scen.p_tx_w[t.device]),
                "rem_bits": float(max(t.rem_bits, 0.0)),
                "tx_start_s": float(t.t_tx_start_s),
                "deadline_s": float(t.deadline_s),
                "C_cycles": float(t.C_cycles),
                "f_hz": float(t.f_hz),
                "tau_dn_s": float(t.tau_dn_s),
            }
            for t in self.active_uplinks
        ]

        n_sb = max(1, int(self.cfg.n_subbands))
        devices_obs = [
            {
                "device": i,
                "f_loc_hz": float(self.scen.f_loc_hz[i]),
                "cpu_used_hz": float(self.dev_cpu_used[i]),
                "cpu_reserved_hz": float(dev_reserved[i]),
                "p_tx_w": float(self.scen.p_tx_w[i]),
                "radio_busy": bool(i in self.busy_radios),
                "tx_queue_bits": float(tx_queue_bits[i]),
                "subband": int(self.scen.subband[i]) % n_sb,
            }
            for i in range(self.n)
        ]

        servers_obs = [
            {
                "server": j,
                "f_max_hz": float(self.scen.f_ms_max_hz[j]),
                "f_avail_hz": float(self.f_avail_hz[j]),
                "cpu_used_hz": float(self.srv_cpu_used[j]),
                "cpu_reserved_hz": float(srv_reserved[j]),
                "wait_compute_count": int(srv_wait_cnt[j]),
            }
            for j in range(self.m)
        ]

        return {
            "time": float(now),
            "tick_s": float(self.cfg.tick_s),
            "delta_sig_s": float(self.cfg.delta_sig_s),
            "rho_sys": float(self.cfg.rho_sys),
            "f_min_hz": float(self.cfg.f_min_hz),
            "pending": pending_obs,
            "active_uplinks": active_obs,
            "devices": devices_obs,
            "servers": servers_obs,
            "links": {
                "h_hat_dev_ms": self._h_obs,
                "bandwidth_hz": float(self.scen.phys.bandwidth_hz),
                "noise_w": float(self.scen.phys.noise_w),
                "orthogonal": bool(self.cfg.orthogonal),
                "n_subbands": n_sb,
            },
        }

    def _compute_uplink_rate(self, t: _RuntimeTask) -> float:
        i = t.device
        j = int(t.server)
        sig = float(self.scen.p_tx_w[i] * self.scen.h_dev_ms[i, j])
        interf = 0.0
        if not self.cfg.orthogonal:
            n_sb = max(1, int(self.cfg.n_subbands))
            sb_i = int(self.scen.subband[i]) % n_sb
            for u in self.active_uplinks:
                if u.tid == t.tid:
                    continue
                if (int(self.scen.subband[u.device]) % n_sb) != sb_i:
                    continue
                interf += float(self.scen.p_tx_w[u.device] * self.scen.h_dev_ms[u.device, j])
        return shannon_rate(self.scen.phys.bandwidth_hz, sig, self.scen.phys.noise_w, interf)

    def _advance_uplinks(self, now: float) -> None:
        dt = max(0.0, float(now) - self._last_reflow_t)
        if dt > 0.0 and self.active_uplinks:
            for t in self.active_uplinks:
                bits = t.cur_rate_bps * dt
                t.integrated_bits += bits
                t.rem_bits = max(0.0, t.rem_bits - bits)
                self.energy_tx_j += float(self.scen.p_tx_w[t.device]) * dt
        self._last_reflow_t = float(now)

    def _recompute_and_schedule_uplinks(self, now: float) -> None:
        if not self.active_uplinks:
            return
        self.reflow_count += 1
        for t in self.active_uplinks:
            rate = self._compute_uplink_rate(t)
            t.cur_rate_bps = rate
            t.tx_ver += 1
            rem_t = max(t.rem_bits, 0.0) / rate
            self._push(now + rem_t, _PRIO_FINISH, "tx_end", (t.tid, t.tx_ver))

    def _handle_tick(self, tick_idx: int, policy: Callable[[dict], dict]) -> None:
        now = self.now
        self._reveal_arrivals(now)
        pending = [t for t in self.tasks if t.state == "pending"]

        next_idx = tick_idx + 1
        if pending:
            obs = self.observe(now)
            t0 = time.perf_counter()
            out = policy(obs)
            wall_s = time.perf_counter() - t0
            self.policy_calls += 1
            self.decision_latencies_ms.append(wall_s * 1000.0)

            if self.cfg.latency_mode == "wall":
                lag = wall_s
            elif self.cfg.latency_mode == "fixed":
                lag = float(self.cfg.fixed_latency_s)
            elif self.cfg.latency_mode == "zero":
                lag = 0.0
            else:
                raise ValueError(f"Unknown latency_mode: {self.cfg.latency_mode}")

            t_exec = now + lag + float(self.cfg.delta_sig_s)
            # Mark valid decisions into actuating states immediately; unmentioned stay pending ("defer").
            staged = self._stage_decisions(out, t_exec)
            if staged:
                self._push(t_exec, _PRIO_ACTUATE, "actuate", (staged, t_exec))

            next_nom = (tick_idx + 1) * self.cfg.tick_s
            if t_exec > next_nom + 1e-12:
                next_idx = int(math.ceil((t_exec - 1e-12) / self.cfg.tick_s))
                self.skipped_ticks += max(0, next_idx - (tick_idx + 1))

        nxt_t = next_idx * self.cfg.tick_s
        if nxt_t <= self.horizon + 1e-9:
            self._push(nxt_t, _PRIO_TICK, "tick", next_idx)

    def _stage_decisions(self, out: object, t_exec: float) -> list[tuple[str, str, Optional[int], Optional[float]]]:
        if not isinstance(out, dict) or not isinstance(out.get("actions"), list):
            self.illegal_actions += 1
            return []
        seen: set[str] = set()
        staged: list[tuple[str, str, Optional[int], Optional[float]]] = []
        for act in out["actions"]:
            if not isinstance(act, dict):
                self.illegal_actions += 1
                continue
            tid = act.get("task")
            if not isinstance(tid, str) or tid not in self.by_id or tid in seen:
                self.illegal_actions += 1
                continue
            t = self.by_id[tid]
            if t.state != "pending":
                self.illegal_actions += 1
                continue
            seen.add(tid)
            mode = act.get("mode", "defer")
            if mode == "defer":
                continue
            if mode == "drop":
                t.state = "actuating_drop"
                t.mode = "drop"
                t.t_exec_s = float(t_exec)
                staged.append((tid, "drop", None, None))
                continue
            if mode not in ("local", "edge"):
                self.illegal_actions += 1
                continue
            try:
                f_hz = float(act.get("f_hz"))
            except (TypeError, ValueError):
                self.illegal_actions += 1
                continue
            if not math.isfinite(f_hz) or f_hz < self.cfg.f_min_hz - 1e-3:
                self.illegal_actions += 1
                continue
            if mode == "local":
                cap = float(self.scen.f_loc_hz[t.device])
                if f_hz > cap + 1.0:
                    self.illegal_actions += 1
                    continue
                f_hz = min(max(f_hz, self.cfg.f_min_hz), cap)
                t.state = "actuating_local"
                t.mode = "local"
                t.server = None
                t.f_hz = float(f_hz)
                t.t_exec_s = float(t_exec)
                staged.append((tid, "local", None, float(f_hz)))
            else:
                srv = act.get("server")
                if not isinstance(srv, (int, np.integer)) or isinstance(srv, bool):
                    self.illegal_actions += 1
                    continue
                srv_idx = int(srv)
                if not (0 <= srv_idx < self.m):
                    self.illegal_actions += 1
                    continue
                cap = float(self.f_avail_hz[srv_idx])
                if f_hz > cap + 1.0:
                    self.illegal_actions += 1
                    continue
                f_hz = min(max(f_hz, self.cfg.f_min_hz), cap)
                t.state = "actuating_edge"
                t.mode = "edge"
                t.server = srv_idx
                t.f_hz = float(f_hz)
                t.t_exec_s = float(t_exec)
                self.n_edge_granted += 1
                staged.append((tid, "edge", srv_idx, float(f_hz)))
        return staged

    def _handle_actuate(self, payload: tuple[list, float]) -> bool:
        staged, _t_exec = payload
        changed_uplinks = False
        for tid, mode, srv_idx, f_hz in staged:
            t = self.by_id[tid]
            if t.state in ("timeout", "done", "dropped"):
                continue
            if mode == "drop":
                t.state = "dropped"
                self.n_dropped += 1
            elif mode == "local":
                t.state = "wait_local"
                self.wait_local_q[t.device].append(t)
            elif mode == "edge":
                t.state = "tx_queue"
                self.tx_queue[t.device].append(t)
        return changed_uplinks

    def _dispatch(self, now: float) -> bool:
        changed_uplinks = False
        # 1. Dispatch half-duplex uplinks per free radio in EDF order
        for i in range(self.n):
            if i in self.busy_radios or not self.tx_queue[i]:
                continue
            self.tx_queue[i].sort(key=lambda t: (t.deadline_s, t.arrival_s, t.k))
            while self.tx_queue[i]:
                t = self.tx_queue[i].pop(0)
                if t.state != "tx_queue":
                    continue
                self.busy_radios.add(i)
                t.state = "tx_up"
                t.t_tx_start_s = float(now)
                t.rem_bits = float(t.D_bits)
                t.integrated_bits = 0.0
                self.active_uplinks.append(t)
                changed_uplinks = True
                break

        # 2. Dispatch local CPU pools in EDF order
        for i in range(self.n):
            if not self.wait_local_q[i]:
                continue
            self.wait_local_q[i].sort(key=lambda t: (t.deadline_s, t.arrival_s, t.k))
            rem_q: list[_RuntimeTask] = []
            for t in self.wait_local_q[i]:
                if t.state != "wait_local":
                    continue
                if self.dev_cpu_used[i] + float(t.f_hz) <= float(self.scen.f_loc_hz[i]) + 1e-6:
                    self.dev_cpu_used[i] = min(float(self.scen.f_loc_hz[i]), self.dev_cpu_used[i] + float(t.f_hz))
                    t.state = "compute"
                    t.t_cmp_start_s = float(now)
                    t.alpha_exec = self.scen.phys.alpha_dev
                    dur = t.C_cycles / float(t.f_hz)
                    self._push(now + dur, _PRIO_FINISH, "compute_end", t.tid)
                else:
                    rem_q.append(t)
            self.wait_local_q[i] = rem_q

        # 3. Dispatch server CPU pools in EDF order (hard rho_sys * F_max cap)
        for j in range(self.m):
            if not self.wait_compute_q[j]:
                continue
            self.wait_compute_q[j].sort(key=lambda t: (t.deadline_s, t.arrival_s, t.k))
            rem_q = []
            for t in self.wait_compute_q[j]:
                if t.state != "wait_compute":
                    continue
                if self.srv_cpu_used[j] + float(t.f_hz) <= float(self.f_avail_hz[j]) + 1e-6:
                    self.srv_cpu_used[j] = min(float(self.f_avail_hz[j]), self.srv_cpu_used[j] + float(t.f_hz))
                    t.state = "compute"
                    t.t_cmp_start_s = float(now)
                    t.alpha_exec = self.scen.phys.alpha_ms
                    dur = t.C_cycles / float(t.f_hz)
                    self._push(now + dur, _PRIO_FINISH, "compute_end", t.tid)
                else:
                    rem_q.append(t)
            self.wait_compute_q[j] = rem_q

        return changed_uplinks

    def _handle_tx_end(self, payload: tuple[str, int], now: float) -> bool:
        tid, ver = payload
        t = self.by_id[tid]
        if t.state != "tx_up" or t.tx_ver != ver:
            return False
        rel_err = abs(t.integrated_bits - t.D_bits) / max(t.D_bits, 1.0)
        if rel_err > self.max_bit_rel_err:
            self.max_bit_rel_err = rel_err
        t.delivered_ok = True
        t.rem_bits = 0.0
        t.t_tx_end_s = float(now)
        self.active_uplinks.remove(t)
        self.busy_radios.discard(t.device)
        t.state = "wait_compute"
        self.wait_compute_q[int(t.server)].append(t)
        return True

    def _handle_compute_end(self, tid: str, now: float) -> None:
        t = self.by_id[tid]
        if t.state != "compute":
            return
        t.t_cmp_end_s = float(now)
        dt_cmp = max(0.0, float(now) - float(t.t_cmp_start_s))
        e_cmp = t.alpha_exec * (float(t.f_hz) ** 2) * t.C_cycles
        t.energy_cmp_j = e_cmp
        self.energy_cmp_j += e_cmp

        if t.mode == "local":
            self.dev_cpu_used[t.device] = max(0.0, self.dev_cpu_used[t.device] - float(t.f_hz))
            self.total_loc_cycles += t.C_cycles
            self._finish_task(t, now)
        else:
            j = int(t.server)
            self.srv_cpu_used[j] = max(0.0, self.srv_cpu_used[j] - float(t.f_hz))
            self.total_srv_cycles += t.C_cycles
            t.state = "tx_down"
            self._push(now + t.tau_dn_s, _PRIO_FINISH, "downlink_end", t.tid)

    def _handle_downlink_end(self, tid: str, now: float) -> None:
        t = self.by_id[tid]
        if t.state != "tx_down":
            return
        self._finish_task(t, now)

    def _finish_task(self, t: _RuntimeTask, now: float) -> None:
        t.state = "done"
        t.t_done_s = float(now)
        if now <= t.deadline_s + 1e-9:
            self.n_ontime += 1
        else:
            self.n_timeout += 1
            if t.mode == "edge":
                self.n_edge_missed += 1

    def _handle_deadline(self, tid: str, now: float) -> bool:
        t = self.by_id[tid]
        if t.state in ("done", "timeout", "dropped"):
            return False
        changed_uplinks = False
        prev_state = t.state
        if t.mode == "edge" and prev_state not in ("unrevealed", "pending", "actuating_drop"):
            self.n_edge_missed += 1

        if prev_state == "tx_up":
            self.active_uplinks.remove(t)
            self.busy_radios.discard(t.device)
            changed_uplinks = True
        elif prev_state == "compute":
            dt_cmp = max(0.0, float(now) - float(t.t_cmp_start_s))
            burned = min(t.C_cycles, float(t.f_hz) * dt_cmp)
            e_cmp = t.alpha_exec * (float(t.f_hz) ** 2) * burned
            t.energy_cmp_j = e_cmp
            self.energy_cmp_j += e_cmp
            if t.mode == "local":
                self.dev_cpu_used[t.device] = max(0.0, self.dev_cpu_used[t.device] - float(t.f_hz))
                self.total_loc_cycles += burned
                self.wasted_loc_cycles += burned
            else:
                j = int(t.server)
                self.srv_cpu_used[j] = max(0.0, self.srv_cpu_used[j] - float(t.f_hz))
                self.total_srv_cycles += burned
                self.wasted_srv_cycles += burned
        elif prev_state == "tx_down" and t.mode == "edge":
            self.wasted_srv_cycles += t.C_cycles

        t.state = "timeout"
        self.n_timeout += 1
        return changed_uplinks

    def _assert_hard_invariants(self, now: float) -> None:
        self.hard_invariant_checks += 1
        # 1. Half-duplex NIC invariant
        assert len(self.busy_radios) == len(self.active_uplinks), (
            f"NIC bookkeeping mismatch at t={now}: busy={len(self.busy_radios)} vs active={len(self.active_uplinks)}"
        )
        assert len({t.device for t in self.active_uplinks}) == len(self.active_uplinks), (
            f"Half-duplex NIC violation at t={now}"
        )
        # 2. Hard server and device CPU capacity caps
        assert np.all(self.srv_cpu_used <= self.f_avail_hz + 1e-3), (
            f"Server CPU cap violated at t={now}: {self.srv_cpu_used} vs {self.f_avail_hz}"
        )
        assert np.all(self.srv_cpu_used >= -1e-3), f"Negative server CPU used at t={now}"
        assert np.all(self.dev_cpu_used <= self.scen.f_loc_hz + 1e-3), (
            f"Device CPU cap violated at t={now}: {self.dev_cpu_used} vs {self.scen.f_loc_hz}"
        )
        assert np.all(self.dev_cpu_used >= -1e-3), f"Negative device CPU used at t={now}"
        # 3. Non-negative remaining bits & bit conservation bound
        for t in self.active_uplinks:
            assert t.rem_bits >= -1e-6, f"Negative remaining bits on {t.tid}"
            assert t.cur_rate_bps > 0.0, f"Non-positive rate on {t.tid}"
        assert self.max_bit_rel_err <= 1e-8, f"Bit conservation error {self.max_bit_rel_err} > 1e-8"
        # 4. Strict causality chain checks on active/completed tasks
        for t in self.tasks:
            if t.t_exec_s is not None:
                assert t.t_exec_s >= t.arrival_s - 1e-12, f"Causality fail t_exec < arrival on {t.tid}"
            if t.t_tx_start_s is not None:
                assert t.t_exec_s is not None and t.t_tx_start_s >= t.t_exec_s - 1e-12, (
                    f"Causality fail tx_start < t_exec on {t.tid}"
                )
            if t.t_cmp_start_s is not None:
                if t.mode == "edge":
                    assert t.t_tx_end_s is not None and t.t_cmp_start_s >= t.t_tx_end_s - 1e-12, (
                        f"Causality fail cmp_start < tx_end on {t.tid}"
                    )
                elif t.mode == "local":
                    assert t.t_exec_s is not None and t.t_cmp_start_s >= t.t_exec_s - 1e-12, (
                        f"Causality fail local cmp_start < t_exec on {t.tid}"
                    )

    def run(self, policy: Callable[[dict], dict]) -> dict:
        while self._heap and not self._stop:
            t_ev = self._heap[0][0]
            if t_ev > self.horizon + 1e-9:
                break
            self.now = t_ev
            self._advance_uplinks(t_ev)

            batch = []
            while self._heap and self._heap[0][0] <= t_ev + 1e-13:
                batch.append(heapq.heappop(self._heap))
            batch.sort(key=lambda x: (x[1], x[2]))

            uplink_dirty = False
            for _w, _prio, _seq, kind, payload in batch:
                if kind == "tx_end":
                    uplink_dirty = self._handle_tx_end(payload, t_ev) or uplink_dirty
                elif kind == "compute_end":
                    self._handle_compute_end(payload, t_ev)
                elif kind == "downlink_end":
                    self._handle_downlink_end(payload, t_ev)
                elif kind == "deadline":
                    uplink_dirty = self._handle_deadline(payload, t_ev) or uplink_dirty
                elif kind == "actuate":
                    self._handle_actuate(payload)
                elif kind == "tick":
                    self._handle_tick(payload, policy)
                elif kind == "horizon":
                    self._stop = True
                    break
                else:
                    raise RuntimeError(f"Unknown event {kind}")

            if self._stop:
                break

            if self._dispatch(t_ev):
                uplink_dirty = True

            if uplink_dirty:
                self._recompute_and_schedule_uplinks(t_ev)

            self._assert_hard_invariants(t_ev)

        r_tot = max(1, self.scen.n_tasks)
        lats = self.decision_latencies_ms
        p50 = float(np.quantile(lats, 0.50)) if lats else 0.0
        p99 = float(np.quantile(lats, 0.99)) if lats else 0.0
        miss_rate = float(self.n_edge_missed / max(1, self.n_edge_granted))
        waste_ratio = float(self.wasted_srv_cycles / max(1.0, self.total_srv_cycles))

        return {
            "case_id": self.scen.case_id,
            "family": self.scen.family,
            "n_tasks": int(self.scen.n_tasks),
            "n_ontime": int(self.n_ontime),
            "ontime_rate": float(self.n_ontime / r_tot),
            "n_timeout": int(self.n_timeout),
            "n_dropped": int(self.n_dropped),
            "n_edge_granted": int(self.n_edge_granted),
            "n_edge_missed": int(self.n_edge_missed),
            "server_grant_miss_rate": miss_rate,
            "wasted_srv_cycle_ratio": waste_ratio,
            "skipped_ticks": int(self.skipped_ticks),
            "decision_p50_ms": p50,
            "decision_p99_ms": p99,
            "policy_calls": int(self.policy_calls),
            "illegal_actions": int(self.illegal_actions),
            "max_bit_rel_err": float(self.max_bit_rel_err),
            "hard_invariant_checks": int(self.hard_invariant_checks),
            "energy_tx_j": float(self.energy_tx_j),
            "energy_cmp_j": float(self.energy_cmp_j),
            "status": self.status,
        }


def run_simulation(scen: ScenarioV1, policy: Callable[[dict], dict], cfg: Optional[SimConfig] = None) -> dict:
    """Convenience runner executing one episode on the read-only EdgeUplinkEnv."""
    return EdgeUplinkEnv(scen, cfg).run(policy)
