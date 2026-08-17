"""1D transmission-line (telegrapher's equations) time-domain solver.

Uses a leapfrog / Yee-type finite-difference time-domain (FDTD) scheme on a
lossy LC ladder equivalent of the cable, with the two boundaries (source and
load) closed by lumped R, L, C elements.

Physical boundary conditions couple a *tiny* capacitance (half of one spatial
cell's shunt capacitance) to lumped resistors/inductors whose own RC/RL time
constants can be much faster than the cable's own propagation time. Plain
forward-Euler on those boundary ODEs is numerically unstable there, so the
boundary states are advanced with an *exact* (zero-order-hold) discretization
of their local linear ODE instead -- see `_zoh_discretize`. The interior of
the line (the actual wave propagation) uses plain leapfrog under the usual
CFL condition, which is where almost all of the physics happens anyway.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

C_LIGHT = 299_792_458.0


def _zoh_discretize(A: np.ndarray, B: np.ndarray, dt: float):
    """Exact zero-order-hold discretization of dx/dt = A x + B u.

    Returns (Ad, Bd) such that x[n+1] = Ad @ x[n] + Bd @ u[n], exact for u
    held constant over the step (standard augmented-matrix trick).
    """
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


class Source:
    """Thevenin source: EMF vs(t) in series with resistance Rs and
    inductance Ls (Ls=0 for an ideal op-amp / resistor source)."""

    def __init__(self, vs_func, Rs: float, Ls: float = 0.0):
        self.vs_func = vs_func
        self.Rs = Rs
        self.Ls = Ls

    def prepare(self, dt: float, Ch: float):
        self._dt = dt
        self._Ch = Ch
        if self.Ls > 0:
            A = np.array([[-self.Rs / self.Ls, -1.0 / self.Ls],
                          [1.0 / Ch, 0.0]])
            B = np.array([[1.0 / self.Ls, 0.0],
                          [0.0, -1.0 / Ch]])
            self._Ad, self._Bd = _zoh_discretize(A, B, dt)
            self.Is = 0.0
        else:
            self.Is = None

    def step(self, V0: float, I0: float, t_new: float) -> float:
        vs = self.vs_func(t_new)
        if self.Ls > 0:
            u = np.array([vs, I0])
            x = np.array([self.Is, V0])
            x_new = self._Ad @ x + self._Bd @ u
            self.Is = x_new[0]
            return x_new[1]
        if self.Rs <= 0:
            return vs
        tau = self.Rs * self._Ch
        Vss = vs - I0 * self.Rs
        return Vss + (V0 - Vss) * np.exp(-self._dt / tau)


class ResistiveLoad:
    """Simple resistor to ground (e.g. a high-Z DAQ input)."""

    def __init__(self, R: float):
        self.R = R

    def prepare(self, dt: float, Ch: float):
        self._dt = dt
        self._Ch = Ch

    def step(self, VN: float, I_last: float, t_new: float):
        tau = self.R * self._Ch
        Vss = I_last * self.R
        VN_new = Vss + (VN - Vss) * np.exp(-self._dt / tau)
        return VN_new, {"Va": VN_new, "Vout": VN_new}


class SeriesRCLoad:
    """R in series with C to ground; the RC low-pass filter.

    The cable's end (top of R) is node `Va`; the tap between R and C
    (the actual measurement point) is `Vb`.
    """

    def __init__(self, R: float, C: float):
        self.R = R
        self.C = C

    def prepare(self, dt: float, Ch: float):
        R, C = self.R, self.C
        A = np.array([[-1.0 / (R * Ch), 1.0 / (R * Ch)],
                      [1.0 / (R * C), -1.0 / (R * C)]])
        B = np.array([[1.0 / Ch], [0.0]])
        self._Ad, self._Bd = _zoh_discretize(A, B, dt)
        self.Vb = 0.0

    def step(self, VN: float, I_last: float, t_new: float):
        u = np.array([I_last])
        x = np.array([VN, self.Vb])
        x_new = self._Ad @ x + self._Bd @ u
        VN_new, self.Vb = x_new[0], x_new[1]
        return VN_new, {"Va": VN_new, "Vout": self.Vb}


class TLineFDTD:
    """Leapfrog FDTD solver for a lossy transmission line, optionally built
    from several concatenated segments of different characteristic
    impedance/velocity (e.g. a short lead-wire stub spliced onto the main
    coax). Segments share one uniform spatial grid; the impedance step at a
    segment junction produces a real, physical partial reflection there --
    no extra boundary machinery is needed for it, only the two true ends
    (source and load) get the lumped-element treatment.
    """

    def __init__(self, Z0: float = None, vf: float = None, length: float = None,
                 source: Source = None, load=None, n_seg: int = 200,
                 cfl: float = 0.9, atten_db_per_m: float = 0.0, segments=None):
        if segments is None:
            segments = [dict(length=length, Z0=Z0, vf=vf, atten_db_per_m=atten_db_per_m)]
        self.segments = segments
        self.length = sum(s["length"] for s in segments)

        self.N = n_seg
        self.dx = self.length / n_seg
        self.x = np.linspace(0.0, self.length, n_seg + 1)

        # per-branch (cell) material properties, assigned by which segment
        # each branch's midpoint falls into
        branch_mid = self.x[:-1] + self.dx / 2.0
        seg_end = np.cumsum([s["length"] for s in segments])
        seg_start = np.concatenate([[0.0], seg_end[:-1]])
        Lp = np.empty(n_seg)
        Cp = np.empty(n_seg)
        Rp = np.empty(n_seg)
        v_of_branch = np.empty(n_seg)
        self.junctions = list(seg_end[:-1])  # internal segment boundaries (for plotting)
        for s, x0, x1 in zip(segments, seg_start, seg_end):
            mask = (branch_mid >= x0) & (branch_mid < x1 + 1e-12)
            v_s = s["vf"] * C_LIGHT
            Lp[mask] = s["Z0"] / v_s
            Cp[mask] = 1.0 / (s["Z0"] * v_s)
            alpha_np_per_m = s.get("atten_db_per_m", 0.0) * 0.115_129_25
            Rp[mask] = 2.0 * alpha_np_per_m * s["Z0"]
            v_of_branch[mask] = v_s
        self.Lp, self.Cp, self.Rp = Lp, Cp, Rp
        self.Z0 = segments[0]["Z0"]     # characteristic impedance at the source end
        self.Z0_load = segments[-1]["Z0"]  # characteristic impedance at the load end
        self.v = v_of_branch[0]          # velocity at the source end (used for CFL display etc.)

        v_max = v_of_branch.max()
        self.dt = cfl * self.dx / v_max

        self.V = np.zeros(n_seg + 1)
        self.I = np.zeros(n_seg)
        self.t = 0.0

        self.source = source
        self.load = load
        Ch0 = Cp[0] * self.dx / 2.0
        ChN = Cp[-1] * self.dx / 2.0
        self.source.prepare(self.dt, Ch0)
        self.load.prepare(self.dt, ChN)

        self._Lseg = Lp * self.dx
        self._Rseg = Rp * self.dx
        self._Cseg_interior = 0.5 * self.dx * (Cp[:-1] + Cp[1:])

    def step(self):
        dt = self.dt
        V, I = self.V, self.I

        # branch currents (interior wave update)
        I += dt * ((V[:-1] - V[1:]) - self._Rseg * I) / self._Lseg

        # interior node voltages (half-cell capacitance from each side --
        # reduces to the uniform-line formula when both sides match)
        V[1:-1] += dt * (I[:-1] - I[1:]) / self._Cseg_interior

        t_new = self.t + dt
        V[0] = self.source.step(V[0], I[0], t_new)
        V[-1], load_out = self.load.step(V[-1], I[-1], t_new)

        self.t = t_new
        return load_out

    def run(self, t_end: float, record_every: int = 1, snapshot_every: int = 0):
        """Advance until t_end. Returns dict of recorded time series and,
        if snapshot_every>0, periodic full spatial snapshots for animation.
        """
        n_steps = int(np.ceil(t_end / self.dt))
        t_hist, va_hist, vout_hist = [], [], []
        snap_t, snap_V = [], []
        for k in range(n_steps):
            out = self.step()
            if k % record_every == 0:
                t_hist.append(self.t)
                va_hist.append(out["Va"])
                vout_hist.append(out["Vout"])
            if snapshot_every and k % snapshot_every == 0:
                snap_t.append(self.t)
                snap_V.append(self.V.copy())
        result = {
            "t": np.array(t_hist),
            "Va": np.array(va_hist),
            "Vout": np.array(vout_hist),
        }
        if snapshot_every:
            result["snap_t"] = np.array(snap_t)
            result["snap_V"] = np.array(snap_V)
        return result
