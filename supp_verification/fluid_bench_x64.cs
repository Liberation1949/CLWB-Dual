using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;

namespace ClwbRevisionAudit
{
    internal struct DeviceInfo
    {
        public double Ptx;
        public int Subband;
        public double FLoc;
        public double CpuRes;
    }

    internal struct ServerInfo
    {
        public double FAvail;
        public double CpuRes;
    }

    internal struct ActiveLink
    {
        public string Id;
        public int Device;
        public int Server;
        public double Ptx;
        public double RemBits;
        public double Deadline;
        public double Cycles;
        public double FLock;
        public double TauDn;
    }

    internal struct TaskGrant
    {
        public string Id;
        public int Device;
        public int Server;
        public double DBits;
        public double Deadline;
        public double Cycles;
        public double TauDn;
        public bool CanDefer;
        public int EdfOrder;
    }

    internal struct VerifyResult
    {
        public bool Feasible;
        public double[] TxEndCand; // indexed by candidate index in input list
    }

    internal sealed class Snapshot
    {
        public double Now;
        public double TAct;
        public double TickS;
        public double DeltaSig;
        public double Bw;
        public double Noise;
        public bool Ortho;
        public int NSb;
        public double FMin;
        public int NDev;
        public int NSrv;
        public DeviceInfo[] Devices;
        public ServerInfo[] Servers;
        public double[,] HMat;
        public ActiveLink[] Active;
        public TaskGrant[] PendingSfa;
        public TaskGrant[] RankedClwb;

        private static string NextLine(StreamReader sr)
        {
            string line;
            while ((line = sr.ReadLine()) != null)
            {
                line = line.Trim();
                if (line.Length > 0) return line;
            }
            throw new EndOfStreamException("Unexpected EOF in snapshot");
        }

        public static Snapshot Load(string path)
        {
            var ci = CultureInfo.InvariantCulture;
            using (var sr = new StreamReader(path))
            {
                string[] p = NextLine(sr).Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                var s = new Snapshot
                {
                    Now = double.Parse(p[0], ci),
                    TAct = double.Parse(p[1], ci),
                    TickS = double.Parse(p[2], ci),
                    DeltaSig = double.Parse(p[3], ci),
                    Bw = double.Parse(p[4], ci),
                    Noise = double.Parse(p[5], ci),
                    Ortho = int.Parse(p[6], ci) != 0,
                    NSb = Math.Max(1, int.Parse(p[7], ci)),
                    FMin = double.Parse(p[8], ci),
                    NDev = int.Parse(p[9], ci),
                    NSrv = int.Parse(p[10], ci),
                };
                int nAct = int.Parse(p[11], ci);
                int nPend = int.Parse(p[12], ci);
                int nCand = int.Parse(p[13], ci);

                s.Devices = new DeviceInfo[s.NDev];
                for (int i = 0; i < s.NDev; i++)
                {
                    p = NextLine(sr).Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                    s.Devices[i] = new DeviceInfo
                    {
                        Ptx = double.Parse(p[0], ci),
                        Subband = int.Parse(p[1], ci) % s.NSb,
                        FLoc = double.Parse(p[2], ci),
                        CpuRes = double.Parse(p[3], ci)
                    };
                }

                s.Servers = new ServerInfo[s.NSrv];
                for (int j = 0; j < s.NSrv; j++)
                {
                    p = NextLine(sr).Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                    s.Servers[j] = new ServerInfo
                    {
                        FAvail = double.Parse(p[0], ci),
                        CpuRes = double.Parse(p[1], ci)
                    };
                }

                s.HMat = new double[s.NDev, s.NSrv];
                for (int i = 0; i < s.NDev; i++)
                {
                    p = NextLine(sr).Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                    for (int j = 0; j < s.NSrv; j++)
                        s.HMat[i, j] = double.Parse(p[j], ci);
                }

                s.Active = new ActiveLink[nAct];
                for (int a = 0; a < nAct; a++)
                {
                    p = NextLine(sr).Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                    s.Active[a] = new ActiveLink
                    {
                        Id = p[0],
                        Device = int.Parse(p[1], ci),
                        Server = int.Parse(p[2], ci),
                        Ptx = double.Parse(p[3], ci),
                        RemBits = double.Parse(p[4], ci),
                        Deadline = double.Parse(p[5], ci),
                        Cycles = double.Parse(p[6], ci),
                        FLock = double.Parse(p[7], ci),
                        TauDn = double.Parse(p[8], ci)
                    };
                }

                s.PendingSfa = new TaskGrant[nPend];
                for (int k = 0; k < nPend; k++)
                {
                    p = NextLine(sr).Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                    s.PendingSfa[k] = new TaskGrant
                    {
                        Id = p[0],
                        Device = int.Parse(p[1], ci),
                        Server = int.Parse(p[2], ci),
                        DBits = double.Parse(p[3], ci),
                        Deadline = double.Parse(p[4], ci),
                        Cycles = double.Parse(p[5], ci),
                        TauDn = double.Parse(p[6], ci),
                        CanDefer = int.Parse(p[7], ci) != 0,
                        EdfOrder = int.Parse(p[8], ci)
                    };
                }

                s.RankedClwb = new TaskGrant[nCand];
                for (int k = 0; k < nCand; k++)
                {
                    p = NextLine(sr).Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                    s.RankedClwb[k] = new TaskGrant
                    {
                        Id = p[0],
                        Device = int.Parse(p[1], ci),
                        Server = int.Parse(p[2], ci),
                        DBits = double.Parse(p[3], ci),
                        Deadline = double.Parse(p[4], ci),
                        Cycles = double.Parse(p[5], ci),
                        TauDn = double.Parse(p[6], ci),
                        CanDefer = int.Parse(p[7], ci) != 0,
                        EdfOrder = int.Parse(p[8], ci)
                    };
                }
                return s;
            }
        }
    }

    internal sealed class EdfComparer : IComparer<TaskGrant>
    {
        public int Compare(TaskGrant a, TaskGrant b)
        {
            return a.EdfOrder.CompareTo(b.EdfOrder);
        }
    }

    internal sealed class FluidWorkspace
    {
        private const double InvLn2 = 1.44269504088896340736; // 1 / ln(2)
        private static readonly IComparer<TaskGrant> EdfCmp = new EdfComparer();
        private readonly Snapshot _s;
        private readonly int _maxK;

        // Reusable preallocated flat buffers (zero GC pressure inside rollout loop)
        private readonly bool[] _isAct;
        private readonly int[] _dev;
        private readonly int[] _srv;
        private readonly int[] _sb;
        private readonly double[] _ptx;
        private readonly double[] _sig;
        private readonly double[] _rem;
        private readonly double[] _dl;
        private readonly double[] _cyc;
        private readonly double[] _flock;
        private readonly double[] _tdn;
        private readonly double[] _radM; // flattened k * M
        private readonly double[] _tTxEnd;
        private readonly int[] _origPos;
        private readonly bool[] _running;
        private readonly int[] _runIdx;
        private readonly double[] _rates;
        private readonly bool[] _busyDev;
        private readonly int[] _devQHead; // single queued flow or linked list per device
        private readonly int[] _nextOnDev;
        private readonly int[] _devQTail;
        private readonly double[] _pSbSrv;
        private readonly double[] _srvAdded;
        private readonly TaskGrant[] _sortedBuf;
        private readonly double[] _candTxEndBuf;

        public FluidWorkspace(Snapshot s)
        {
            _s = s;
            _maxK = s.Active.Length + Math.Max(s.PendingSfa.Length, s.RankedClwb.Length) + 4;
            _isAct = new bool[_maxK];
            _dev = new int[_maxK];
            _srv = new int[_maxK];
            _sb = new int[_maxK];
            _ptx = new double[_maxK];
            _sig = new double[_maxK];
            _rem = new double[_maxK];
            _dl = new double[_maxK];
            _cyc = new double[_maxK];
            _flock = new double[_maxK];
            _tdn = new double[_maxK];
            _radM = new double[_maxK * s.NSrv];
            _tTxEnd = new double[_maxK];
            _origPos = new int[_maxK];
            _running = new bool[_maxK];
            _runIdx = new int[_maxK];
            _rates = new double[_maxK];
            _busyDev = new bool[s.NDev];
            _devQHead = new int[s.NDev];
            _devQTail = new int[s.NDev];
            _nextOnDev = new int[_maxK];
            _pSbSrv = new double[s.NSb * s.NSrv];
            _srvAdded = new double[s.NSrv];
            _sortedBuf = new TaskGrant[_maxK];
            _candTxEndBuf = new double[_maxK];
        }

        private void EnqueueDev(int di, int idx)
        {
            _nextOnDev[idx] = -1;
            if (_devQHead[di] == -1)
            {
                _devQHead[di] = idx;
                _devQTail[di] = idx;
            }
            else
            {
                _nextOnDev[_devQTail[di]] = idx;
                _devQTail[di] = idx;
            }
        }

        private void DispatchDeviceQueues(int nDev)
        {
            for (int di = 0; di < nDev; di++)
            {
                if (!_busyDev[di] && _devQHead[di] != -1)
                {
                    int nxt = _devQHead[di];
                    _devQHead[di] = _nextOnDev[nxt];
                    if (_devQHead[di] == -1) _devQTail[di] = -1;
                    _running[nxt] = true;
                    _busyDev[di] = true;
                }
            }
        }

        public bool Verify(TaskGrant[] cands, int candCount, double actLagS, out double[] candTxEnds, out double[] fullFlowTxEnds)
        {
            int nAct = _s.Active.Length;
            int nDev = _s.NDev;
            int m = _s.NSrv;
            int nSb = _s.NSb;
            double now = _s.Now;
            double tAct = now + _s.DeltaSig + actLagS;
            bool relImmediate = tAct <= now + 1e-12;
            int kTot = nAct + candCount;

            candTxEnds = _candTxEndBuf;
            fullFlowTxEnds = _tTxEnd;
            if (nAct == 0 && candCount == 0)
            {
                return true;
            }

            for (int c = 0; c < candCount; c++)
            {
                _sortedBuf[c] = cands[c];
                _origPos[c] = c;
            }
            Array.Sort(_sortedBuf, _origPos, 0, candCount, EdfCmp);

            for (int di = 0; di < nDev; di++)
            {
                _busyDev[di] = false;
                _devQHead[di] = -1;
                _devQTail[di] = -1;
            }

            for (int a = 0; a < nAct; a++)
            {
                ActiveLink u = _s.Active[a];
                int di = u.Device;
                int ji = u.Server;
                _isAct[a] = true;
                _dev[a] = di;
                _srv[a] = ji;
                _sb[a] = _s.Devices[di].Subband;
                _ptx[a] = u.Ptx;
                _sig[a] = u.Ptx * _s.HMat[di, ji];
                _rem[a] = u.RemBits > 0.0 ? u.RemBits : 0.0;
                _dl[a] = u.Deadline;
                _cyc[a] = u.Cycles;
                _flock[a] = u.FLock;
                _tdn[a] = u.TauDn;
                _tTxEnd[a] = double.PositiveInfinity;
                _running[a] = true;
                _busyDev[di] = true;
                int rowOff = a * m;
                for (int j = 0; j < m; j++) _radM[rowOff + j] = u.Ptx * _s.HMat[di, j];
            }

            for (int c = 0; c < candCount; c++)
            {
                int idx = nAct + c;
                TaskGrant g = _sortedBuf[c];
                int di = g.Device;
                int ji = g.Server;
                _isAct[idx] = false;
                _dev[idx] = di;
                _srv[idx] = ji;
                _sb[idx] = _s.Devices[di].Subband;
                double ptx = _s.Devices[di].Ptx;
                _ptx[idx] = ptx;
                _sig[idx] = ptx * _s.HMat[di, ji];
                _rem[idx] = g.DBits;
                _dl[idx] = g.Deadline;
                _cyc[idx] = g.Cycles;
                _flock[idx] = 0.0;
                _tdn[idx] = g.TauDn;
                _tTxEnd[idx] = double.PositiveInfinity;
                _running[idx] = false;
                int rowOff = idx * m;
                for (int j = 0; j < m; j++) _radM[rowOff + j] = ptx * _s.HMat[di, j];
                if (relImmediate) EnqueueDev(di, idx);
            }

            if (relImmediate) DispatchDeviceQueues(nDev);

            double tCur = now;
            bool released = relImmediate;
            double bw = _s.Bw;
            double noise = _s.Noise;
            bool ortho = _s.Ortho;

            while (true)
            {
                int runCnt = 0;
                for (int idx = 0; idx < kTot; idx++)
                {
                    if (_running[idx]) _runIdx[runCnt++] = idx;
                }

                if (runCnt == 0)
                {
                    if (released) break;
                    tCur = tAct;
                    released = true;
                    for (int idx = nAct; idx < kTot; idx++) EnqueueDev(_dev[idx], idx);
                    DispatchDeviceQueues(nDev);
                    continue;
                }

                double minDtEnd = double.PositiveInfinity;
                if (ortho)
                {
                    double denomN = noise > 1e-30 ? noise : 1e-30;
                    for (int r = 0; r < runCnt; r++)
                    {
                        int idx = _runIdx[r];
                        double rate = bw * Math.Log(1.0 + _sig[idx] / denomN) * InvLn2;
                        if (rate < 1e-3) rate = 1e-3;
                        _rates[r] = rate;
                        double dtEnd = _rem[idx] / rate;
                        if (dtEnd < minDtEnd) minDtEnd = dtEnd;
                    }
                }
                else
                {
                    Array.Clear(_pSbSrv, 0, nSb * m);
                    for (int r = 0; r < runCnt; r++)
                    {
                        int idx = _runIdx[r];
                        int sbOff = _sb[idx] * m;
                        int radOff = idx * m;
                        for (int j = 0; j < m; j++)
                        {
                            _pSbSrv[sbOff + j] += _radM[radOff + j];
                        }
                    }
                    for (int r = 0; r < runCnt; r++)
                    {
                        int idx = _runIdx[r];
                        double tot = _pSbSrv[_sb[idx] * m + _srv[idx]];
                        double interf = tot - _sig[idx];
                        if (interf < 0.0) interf = 0.0;
                        double denom = noise + interf;
                        if (denom < 1e-30) denom = 1e-30;
                        double rate = bw * Math.Log(1.0 + _sig[idx] / denom) * InvLn2;
                        if (rate < 1e-3) rate = 1e-3;
                        _rates[r] = rate;
                        double dtEnd = _rem[idx] / rate;
                        if (dtEnd < minDtEnd) minDtEnd = dtEnd;
                    }
                }

                double dtRel = (!released && candCount > 0) ? Math.Max(0.0, tAct - tCur) : double.PositiveInfinity;
                double dt = minDtEnd < dtRel ? minDtEnd : dtRel;

                bool finishedAny = false;
                for (int r = 0; r < runCnt; r++)
                {
                    int idx = _runIdx[r];
                    double nr = _rem[idx] - _rates[r] * dt;
                    if (nr < 0.0) nr = 0.0;
                    _rem[idx] = nr;
                }
                tCur += dt;

                for (int r = 0; r < runCnt; r++)
                {
                    int idx = _runIdx[r];
                    if (_rem[idx] <= 1e-6)
                    {
                        _rem[idx] = 0.0;
                        _tTxEnd[idx] = tCur;
                        _running[idx] = false;
                        _busyDev[_dev[idx]] = false;
                        finishedAny = true;
                    }
                }

                if (!released && Math.Abs(tCur - tAct) <= 1e-12)
                {
                    released = true;
                    for (int idx = nAct; idx < kTot; idx++) EnqueueDev(_dev[idx], idx);
                    DispatchDeviceQueues(nDev);
                }
                else if (finishedAny)
                {
                    DispatchDeviceQueues(nDev);
                }
            }

            // Feasibility check matching Python _check_fluid_feasibility
            for (int c = 0; c < candCount; c++)
            {
                _candTxEndBuf[_origPos[c]] = _tTxEnd[nAct + c];
            }

            for (int idx = 0; idx < kTot; idx++)
            {
                if (double.IsInfinity(_tTxEnd[idx]) || double.IsNaN(_tTxEnd[idx])) return false;
            }

            Array.Clear(_srvAdded, 0, m);
            bool feasible = true;
            double fMin = _s.FMin;
            for (int idx = 0; idx < kTot; idx++)
            {
                double tUp = _tTxEnd[idx];
                if (_isAct[idx])
                {
                    double fLock = _flock[idx] > 1.0 ? _flock[idx] : 1.0;
                    bool ok = (tUp + _cyc[idx] / fLock + _tdn[idx]) <= (_dl[idx] + 1e-9);
                    if (!ok) feasible = false;
                }
                else
                {
                    double slackCmp = _dl[idx] - tUp - _tdn[idx];
                    if (slackCmp <= 1e-9)
                    {
                        feasible = false;
                    }
                    else
                    {
                        double fReq = _cyc[idx] / slackCmp;
                        if (fReq < fMin) fReq = fMin;
                        int j = _srv[idx];
                        _srvAdded[j] += fReq;
                        if (fReq > _s.Servers[j].FAvail * (1.0 - 1e-9)) feasible = false;
                    }
                }
            }

            if (feasible)
            {
                for (int j = 0; j < m; j++)
                {
                    if (_s.Servers[j].CpuRes + _srvAdded[j] > _s.Servers[j].FAvail * (1.0 - 1e-9))
                    {
                        feasible = false;
                        break;
                    }
                }
            }
            return feasible;
        }
    }

    internal static class Program
    {
        private static bool IsAdmissible(Snapshot s, FluidWorkspace ws, TaskGrant[] sub, int count, out int callsAdded)
        {
            callsAdded = 1;
            double[] candTxEnds;
            double[] fullEnds;
            bool ok = ws.Verify(sub, count, 0.0, out candTxEnds, out fullEnds);
            if (!ok) return false;
            if (!s.Ortho)
            {
                double horizon = s.Now + 2.0 * s.TickS;
                for (int i = 0; i < count; i++)
                {
                    if (sub[i].CanDefer && candTxEnds[i] > horizon) return false;
                }
            }
            return true;
        }

        private static int RunClwbSelection(Snapshot s, FluidWorkspace ws, out int verifyCalls, out int admittedCount)
        {
            TaskGrant[] cands = s.RankedClwb;
            int p = cands.Length;
            verifyCalls = 0;
            if (p == 0)
            {
                admittedCount = 0;
                return 0;
            }

            double[] dummyCand;
            double[] dummyFull;
            int c0;
            if (IsAdmissible(s, ws, cands, p, out c0))
            {
                verifyCalls += c0;
                admittedCount = p;
                return admittedCount;
            }
            verifyCalls += c0;

            int lo = 0;
            int hi = p > 0 ? p - 1 : 0;
            TaskGrant[] buf = new TaskGrant[p];
            for (int i = 0; i < p; i++) buf[i] = cands[i];
            while (lo < hi)
            {
                int mid = (lo + hi + 1) / 2;
                int cm;
                bool okM = IsAdmissible(s, ws, buf, mid, out cm);
                verifyCalls += cm;
                if (okM) lo = mid;
                else hi = mid - 1;
            }

            int kept = lo;
            int budgetChecks = 2 * s.NSrv + 2;
            int checks = 0;
            for (int idx = lo; idx < p; idx++)
            {
                if (checks >= budgetChecks) break;
                checks++;
                buf[kept] = cands[idx];
                int ct;
                bool okT = IsAdmissible(s, ws, buf, kept + 1, out ct);
                verifyCalls += ct;
                if (okT) kept++;
            }

            if (kept > 0)
            {
                verifyCalls++;
                bool okFin = ws.Verify(buf, kept, 0.0, out dummyCand, out dummyFull);
                if (!okFin)
                {
                    int flo = 0, fhi = kept;
                    while (flo < fhi)
                    {
                        int fmid = (flo + fhi + 1) / 2;
                        verifyCalls++;
                        if (ws.Verify(buf, fmid, 0.0, out dummyCand, out dummyFull)) flo = fmid;
                        else fhi = fmid - 1;
                    }
                    kept = flo;
                }
            }
            admittedCount = kept;
            return kept;
        }

        private static int RunSfaSelection(Snapshot s, FluidWorkspace ws, out int verifyCalls, out int admittedCount)
        {
            int p = s.PendingSfa.Length;
            int m = s.NSrv;
            double now = s.Now;
            double tExec = now + s.DeltaSig;
            double fMin = s.FMin;

            double[] devRes = new double[s.NDev];
            for (int i = 0; i < s.NDev; i++) devRes[i] = s.Devices[i].CpuRes;

            TaskGrant[] admittedEdge = new TaskGrant[p + 1];
            int admEdgeCnt = 0;
            int admLocCnt = 0;
            verifyCalls = 0;
            double[] candTxEnds;
            double[] dummyFull;

            for (int k = 0; k < p; k++)
            {
                TaskGrant t = s.PendingSfa[k];
                int di = t.Device;
                double dl = t.Deadline;
                double cCyc = t.Cycles;

                bool hasBestEdge = false;
                double bestF = 0.0, bestTfin = 0.0;
                int bestJ = -1;
                TaskGrant bestCand = t;

                for (int j = 0; j < m; j++)
                {
                    TaskGrant cand = t;
                    cand.Server = j;
                    admittedEdge[admEdgeCnt] = cand;
                    verifyCalls++;
                    bool ok = ws.Verify(admittedEdge, admEdgeCnt + 1, 0.0, out candTxEnds, out dummyFull);
                    if (ok)
                    {
                        double tUp = candTxEnds[admEdgeCnt];
                        double slack = dl - tUp - t.TauDn;
                        double fReq = slack > 1e-9 ? Math.Max(fMin, cCyc / slack) : double.PositiveInfinity;
                        double tFin = tUp + cCyc / fReq + t.TauDn;
                        if (!hasBestEdge || fReq < bestF || (fReq == bestF && (tFin < bestTfin || (tFin == bestTfin && j < bestJ))))
                        {
                            hasBestEdge = true;
                            bestF = fReq;
                            bestTfin = tFin;
                            bestJ = j;
                            bestCand = cand;
                        }
                    }
                }

                bool hasBestLoc = false;
                double locFreq = 0.0;
                double remLoc = dl - tExec;
                if (remLoc > 1e-9)
                {
                    double fReqL = Math.Max(fMin, cCyc / remLoc);
                    if (devRes[di] + fReqL <= s.Devices[di].FLoc + 1.0)
                    {
                        hasBestLoc = true;
                        locFreq = fReqL;
                    }
                }

                bool chooseLoc = false;
                if (hasBestLoc && hasBestEdge)
                {
                    double eLoc = 1e-27 * locFreq * locFreq * cCyc;
                    double tUpDur = bestTfin - (cCyc / bestF + t.TauDn) - now;
                    double eEdge = s.Devices[di].Ptx * Math.Max(tUpDur, 0.0);
                    if (eLoc <= eEdge) chooseLoc = true;
                }
                else if (hasBestLoc)
                {
                    chooseLoc = true;
                }

                if (hasBestLoc)
                {
                    double fAlloc = Math.Min(s.Devices[di].FLoc, Math.Max(fMin, locFreq * 1.02));
                    locFreq = fAlloc;
                }

                if (chooseLoc && hasBestLoc)
                {
                    devRes[di] += locFreq;
                    admLocCnt++;
                }
                else if (hasBestEdge)
                {
                    admittedEdge[admEdgeCnt++] = bestCand;
                }
            }
            admittedCount = admEdgeCnt + admLocCnt;
            return admittedCount;
        }

        public static int Main(string[] args)
        {
            if (args.Length < 2)
            {
                Console.Error.WriteLine("Usage: fluid_bench_x64.exe <parity|bench> <snapshot_path> [clwb_reps] [sfa_reps]");
                return 2;
            }
            string mode = args[0];
            Snapshot snap = Snapshot.Load(args[1]);
            var ws = new FluidWorkspace(snap);
            var ci = CultureInfo.InvariantCulture;

            if (mode == "parity")
            {
                double[] candTxEnds;
                double[] dummyFull;
                bool ok = ws.Verify(snap.RankedClwb, snap.RankedClwb.Length, 0.0, out candTxEnds, out dummyFull);
                Console.Write(ok ? "FEAS=1" : "FEAS=0");
                for (int i = 0; i < snap.RankedClwb.Length; i++)
                {
                    Console.Write(" " + candTxEnds[i].ToString("R", ci));
                }
                Console.WriteLine();
                return 0;
            }

            int clwbReps = args.Length >= 3 ? int.Parse(args[2], ci) : 200;
            int sfaReps = args.Length >= 4 ? int.Parse(args[3], ci) : 3;

            int warmC, warmA;
            RunClwbSelection(snap, ws, out warmC, out warmA);
            RunSfaSelection(snap, ws, out warmC, out warmA);

            int clwbCalls = 0, clwbAdm = 0;
            var sw = Stopwatch.StartNew();
            for (int r = 0; r < clwbReps; r++)
            {
                RunClwbSelection(snap, ws, out clwbCalls, out clwbAdm);
            }
            sw.Stop();
            double clwbMs = sw.Elapsed.TotalMilliseconds / clwbReps;

            int sfaCalls = 0, sfaAdm = 0;
            sw.Restart();
            for (int r = 0; r < sfaReps; r++)
            {
                RunSfaSelection(snap, ws, out sfaCalls, out sfaAdm);
            }
            sw.Stop();
            double sfaMs = sw.Elapsed.TotalMilliseconds / sfaReps;

            Console.WriteLine(string.Format(
                ci,
                "CLWB_MS={0:F6} CLWB_CALLS={1} CLWB_ADM={2} SFA_MS={3:F6} SFA_CALLS={4} SFA_ADM={5}",
                clwbMs, clwbCalls, clwbAdm, sfaMs, sfaCalls, sfaAdm
            ));
            return 0;
        }
    }
}
