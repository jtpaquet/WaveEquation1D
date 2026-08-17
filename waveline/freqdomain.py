"""Exact frequency-domain (phasor) transmission-line solver.

Used for two things the time-domain FDTD is a poor/slow tool for:

1. The *exact* steady-state waveform at the load, including for a square
   wave (built from its odd-harmonic Fourier series and the line's transfer
   function at each harmonic) -- this is what "settles to" after the FDTD
   transient dies out, and is much cheaper/exact to compute directly.
2. The spatial standing-wave envelope |V(x)| and phase(x) along the cable,
   which is what makes the effect of impedance matching on phase visible.

Supports a *cascade* of segments of different characteristic impedance
(e.g. a short lead-wire stub spliced onto the main coax) via standard
transmission-line chain analysis: the load-side segment's input impedance
becomes the "load" the next segment upstream sees, iterated back to the
source; forward voltage/current are then propagated the other way to build
the full spatial profile.

Phasor convention: a real time-domain tone A*sin(w t) is represented by the
*real* phasor coefficient A multiplying exp(1j*w*t), and reconstructed via
Im(phasor * exp(1j*w*t)). This matches sin() exactly and lets every
frequency (fundamental or harmonic) be treated identically.
"""
from __future__ import annotations

import numpy as np

C_LIGHT = 299_792_458.0


def gamma_of_f(f, Z0, vf, atten_db_per_m_ref, f_ref):
    """Propagation constant alpha(f) + j*beta(f).

    alpha is scaled ~sqrt(f/f_ref), the usual skin-effect law for coax at
    these frequencies, referenced to `atten_db_per_m_ref` at `f_ref`.
    """
    v = vf * C_LIGHT
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.sqrt(np.abs(f) / f_ref)
    alpha_db_per_m = atten_db_per_m_ref * scale
    alpha_np_per_m = alpha_db_per_m * 0.115_129_25
    beta = 2 * np.pi * f / v
    return alpha_np_per_m + 1j * beta


def _segment_gamma(seg, freq):
    return gamma_of_f(freq, seg["Z0"], seg["vf"],
                       seg.get("atten_db_per_m_ref", 0.0), seg.get("f_ref", 1e6))


def _as_segments(segments, Z0, vf, length, atten_db_per_m_ref, f_ref):
    if segments is not None:
        return segments
    return [dict(length=length, Z0=Z0, vf=vf,
                 atten_db_per_m_ref=atten_db_per_m_ref, f_ref=f_ref)]


def cascade_input_impedance(segments, freq, ZL):
    """Input impedance looking into the whole cascade from the source end,
    at the given (scalar or array) frequency."""
    Z = ZL
    for seg in reversed(segments):
        gamma = _segment_gamma(seg, freq)
        Z0, length = seg["Z0"], seg["length"]
        thl = np.tanh(gamma * length)
        Z = Z0 * (Z + Z0 * thl) / (Z0 + Z * thl)
    return Z


def cascade_solve(segments, freq, Vs, Zs, ZL):
    """Forward/backward traveling-wave phasors for every segment, plus the
    total input impedance/voltage and the voltage at the far (load) end.
    """
    Zin_total = cascade_input_impedance(segments, freq, ZL)
    Vin = Vs * Zin_total / (Zin_total + Zs)
    Iin = Vin / Zin_total
    seg_results = []
    V_cur, I_cur = Vin, Iin
    for seg in segments:
        gamma = _segment_gamma(seg, freq)
        Z0, length = seg["Z0"], seg["length"]
        Vplus = (V_cur + Z0 * I_cur) / 2.0
        Vminus = (V_cur - Z0 * I_cur) / 2.0
        seg_results.append(dict(Z0=Z0, gamma=gamma, length=length,
                                 Vplus=Vplus, Vminus=Vminus))
        ex_m = np.exp(-gamma * length)
        ex_p = np.exp(gamma * length)
        V_end = Vplus * ex_m + Vminus * ex_p
        I_end = (Vplus * ex_m - Vminus * ex_p) / Z0
        V_cur, I_cur = V_end, I_end
    return seg_results, Zin_total, Vin, V_cur  # V_cur here is Va, the true load-end voltage


def cascade_voltage_profile(seg_results, x):
    """V(x) phasor(s) for positions x measured from the start of segment 1.
    Each per-segment Vplus/gamma may be scalar (single frequency) or a 1D
    array over harmonics; output shape follows suit: (len(x),) or
    (len(x), n_harmonics).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    lengths = [s["length"] for s in seg_results]
    seg_ends = np.cumsum(lengths)
    seg_starts = np.concatenate([[0.0], seg_ends[:-1]])
    sample_shape = np.shape(seg_results[0]["Vplus"])
    out = np.zeros((len(x),) + sample_shape, dtype=complex)
    for s, x0, x1 in zip(seg_results, seg_starts, seg_ends):
        mask = (x >= x0 - 1e-9) & (x <= x1 + 1e-9)
        xi = x[mask] - x0
        if sample_shape:
            val = (s["Vplus"][None, :] * np.exp(-s["gamma"][None, :] * xi[:, None]) +
                   s["Vminus"][None, :] * np.exp(s["gamma"][None, :] * xi[:, None]))
        else:
            val = s["Vplus"] * np.exp(-s["gamma"] * xi) + s["Vminus"] * np.exp(s["gamma"] * xi)
        out[mask] = val
    return out


def steady_state_sine(f0, Vs_amp, Zs_of_f, ZL_of_f, Z0, vf, length, x,
                       atten_db_per_m_ref=0.0, f_ref=None, tap_divider=None,
                       segments=None):
    """Steady state for a single-tone sine source of amplitude Vs_amp (V),
    i.e. v_s(t) = Vs_amp*sin(2*pi*f0*t). `segments`, if given, overrides
    Z0/vf/length with a cascade of differently-impedanced sections (ordered
    source -> load); Z0/vf/length are then ignored for propagation (only
    used as fallback if segments is None).

    Returns dict with V(x) phasor profile, Va(t) and (optionally) Vout(t)
    time series over one period, and the reflection coefficients.
    """
    if f_ref is None:
        f_ref = f0
    segs = _as_segments(segments, Z0, vf, length, atten_db_per_m_ref, f_ref)
    Zs = Zs_of_f(f0)
    ZL = ZL_of_f(f0)
    seg_results, Zin, Vin, Va_phasor = cascade_solve(segs, f0, Vs_amp, Zs, ZL)
    Vprofile = cascade_voltage_profile(seg_results, x)
    Gamma_S = (Zs - segs[0]["Z0"]) / (Zs + segs[0]["Z0"])
    Gamma_L = (ZL - segs[-1]["Z0"]) / (ZL + segs[-1]["Z0"])

    t = np.linspace(0, 1.0 / f0, 400, endpoint=False)
    w = 2 * np.pi * f0
    Va_t = np.imag(Va_phasor * np.exp(1j * w * t))
    Vs_t = Vs_amp * np.sin(w * t)

    out = {
        "x": x, "V_x_phasor": Vprofile, "Va_phasor": Va_phasor,
        "t": t, "Vs_t": Vs_t, "Va_t": Va_t,
        "Gamma_S": Gamma_S, "Gamma_L": Gamma_L, "Zin": Zin,
        "f0": f0, "junctions": np.cumsum([s["length"] for s in segs])[:-1],
    }
    if tap_divider is not None:
        Vout_phasor = tap_divider(Va_phasor, f0)
        out["Vout_phasor"] = Vout_phasor
        out["Vout_t"] = np.imag(Vout_phasor * np.exp(1j * w * t))
    return out


def harmonics_from_waveform(vs_func, f0, n_max=101, n_samples=8192):
    """FFT the actual (possibly finite-rise-time) periodic source waveform
    to get its harmonic content, rather than assuming an ideal/infinitely
    sharp edge. Returned phasors use this module's Im(P*exp(jwt)) convention
    (an FFT naturally gives a Re() convention; converted via P = -1j*C).
    """
    T = 1.0 / f0
    t = np.arange(n_samples) / n_samples * T
    v = np.array([vs_func(tt) for tt in t])
    V = np.fft.rfft(v)
    n = np.arange(1, n_max + 1)
    coeff = -1j * (2.0 / n_samples) * V[n]
    freq = n * f0
    # drop negligible harmonics (keeps runtime down without changing results)
    keep = np.abs(coeff) > 1e-6 * np.max(np.abs(coeff))
    return n[keep], freq[keep], coeff[keep]


def steady_state_square(f0, vs_func, Zs_of_f, ZL_of_f, Z0, vf, length, x,
                         atten_db_per_m_ref=0.0, f_ref=None, n_max=101,
                         tap_divider=None, n_time=2000, amplitude=None,
                         segments=None):
    """Steady state for a periodic source waveform vs_func(t) (period 1/f0),
    via its harmonic (FFT) decomposition. See `steady_state_sine` for the
    `segments` cascade option.
    """
    if f_ref is None:
        f_ref = f0
    segs = _as_segments(segments, Z0, vf, length, atten_db_per_m_ref, f_ref)
    n, freq, coeff = harmonics_from_waveform(vs_func, f0, n_max=n_max)
    Zs = Zs_of_f(freq)
    ZL = ZL_of_f(freq)
    seg_results, Zin, Vin, Va_harm = cascade_solve(segs, freq, coeff, Zs, ZL)
    Vx_harm = cascade_voltage_profile(seg_results, x)  # shape (len(x), n_harm)

    t = np.linspace(0, 2.0 / f0, n_time, endpoint=False)
    w = 2 * np.pi * freq
    Va_t = np.imag(Va_harm[None, :] * np.exp(1j * np.outer(t, w))).sum(axis=1)
    tp = np.mod(t, 1.0 / f0)
    Vs_t = np.array([vs_func(tt) for tt in tp])

    Vx_t = np.imag(Vx_harm[:, None, :] * np.exp(1j * np.outer(t, w))[None, :, :]).sum(axis=2)

    Zs0, ZL0 = Zs_of_f(f0), ZL_of_f(f0)
    out = {
        "x": x, "t": t, "Vs_t": Vs_t, "Va_t": Va_t, "Vx_t": Vx_t,
        "n": n, "freq": freq, "Va_harm": Va_harm,
        "Gamma_S": (Zs0 - segs[0]["Z0"]) / (Zs0 + segs[0]["Z0"]),
        "Gamma_L": (ZL0 - segs[-1]["Z0"]) / (ZL0 + segs[-1]["Z0"]),
        "f0": f0, "junctions": np.cumsum([s["length"] for s in segs])[:-1],
    }
    if tap_divider is not None:
        Vout_harm = tap_divider(Va_harm, freq)
        out["Vout_harm"] = Vout_harm
        out["Vout_t"] = np.imag(Vout_harm[None, :] * np.exp(1j * np.outer(t, w))).sum(axis=1)
    return out
