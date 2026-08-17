"""Exact frequency-domain (phasor) transmission-line solver.

Used for two things the time-domain FDTD is a poor/slow tool for:

1. The *exact* steady-state waveform at the load, including for a square
   wave (built from its odd-harmonic Fourier series and the line's transfer
   function at each harmonic) -- this is what "settles to" after the FDTD
   transient dies out, and is much cheaper/exact to compute directly.
2. The spatial standing-wave envelope |V(x)| and phase(x) along the cable,
   which is what makes the effect of impedance matching on phase visible.

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


def line_input_impedance(Z0, gamma, length, ZL):
    thl = np.tanh(gamma * length)
    return Z0 * (ZL + Z0 * thl) / (Z0 + ZL * thl)


def wave_amplitudes(Vs, Zs, ZL, Z0, gamma, length):
    """Forward/backward traveling-wave phasors V+, V- at x=0."""
    Zin = line_input_impedance(Z0, gamma, length, ZL)
    Vin = Vs * Zin / (Zin + Zs)
    Gamma_L = (ZL - Z0) / (ZL + Z0)
    Vplus = Vin / (1.0 + Gamma_L * np.exp(-2 * gamma * length))
    Vminus = Gamma_L * np.exp(-2 * gamma * length) * Vplus
    return Vplus, Vminus, Vin, Zin, Gamma_L


def voltage_profile(x, Vplus, Vminus, gamma):
    """V(x) phasor for an array of positions x (broadcasts over gamma too)."""
    return Vplus * np.exp(-gamma * x) + Vminus * np.exp(gamma * x)


def steady_state_sine(f0, Vs_amp, Zs_of_f, ZL_of_f, Z0, vf, length, x,
                       atten_db_per_m_ref=0.0, f_ref=None, tap_divider=None):
    """Steady state for a single-tone sine source of amplitude Vs_amp (V),
    i.e. v_s(t) = Vs_amp*sin(2*pi*f0*t).

    Returns dict with V(x) phasor profile, Va(t) and (optionally) Vout(t)
    time series over one period, and the reflection coefficients.
    """
    if f_ref is None:
        f_ref = f0
    gamma = gamma_of_f(f0, Z0, vf, atten_db_per_m_ref, f_ref)
    Zs = Zs_of_f(f0)
    ZL = ZL_of_f(f0)
    Vplus, Vminus, Vin, Zin, Gamma_L = wave_amplitudes(Vs_amp, Zs, ZL, Z0, gamma, length)
    Vprofile = voltage_profile(x, Vplus, Vminus, gamma)
    Va_phasor = voltage_profile(length, Vplus, Vminus, gamma)
    Gamma_S = (Zs - Z0) / (Zs + Z0)

    t = np.linspace(0, 1.0 / f0, 400, endpoint=False)
    w = 2 * np.pi * f0
    Va_t = np.imag(Va_phasor * np.exp(1j * w * t))
    Vs_t = Vs_amp * np.sin(w * t)

    out = {
        "x": x, "V_x_phasor": Vprofile, "Va_phasor": Va_phasor,
        "t": t, "Vs_t": Vs_t, "Va_t": Va_t,
        "Gamma_S": Gamma_S, "Gamma_L": Gamma_L, "Zin": Zin,
        "f0": f0,
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
                         tap_divider=None, n_time=2000, amplitude=None):
    """Steady state for a periodic source waveform vs_func(t) (period 1/f0),
    via its harmonic (FFT) decomposition. `amplitude` is only used to build
    the displayed ideal-edge reference Vs_t if given; otherwise vs_func
    itself is sampled for Vs_t.
    """
    if f_ref is None:
        f_ref = f0
    n, freq, coeff = harmonics_from_waveform(vs_func, f0, n_max=n_max)
    gamma = gamma_of_f(freq, Z0, vf, atten_db_per_m_ref, f_ref)
    Zs = Zs_of_f(freq)
    ZL = ZL_of_f(freq)
    Vplus, Vminus, Vin, Zin, Gamma_L = wave_amplitudes(coeff, Zs, ZL, Z0, gamma, length)

    Va_harm = voltage_profile(length, Vplus, Vminus, gamma)
    Vx_harm = voltage_profile(x[:, None], Vplus[None, :], Vminus[None, :], gamma[None, :])

    t = np.linspace(0, 2.0 / f0, n_time, endpoint=False)
    w = 2 * np.pi * freq
    Va_t = np.imag(Va_harm[None, :] * np.exp(1j * np.outer(t, w))).sum(axis=1)
    tp = np.mod(t, 1.0 / f0)
    Vs_t = np.array([vs_func(tt) for tt in tp])

    Vx_t = np.imag(Vx_harm[:, None, :] * np.exp(1j * np.outer(t, w))[None, :, :]).sum(axis=2)

    out = {
        "x": x, "t": t, "Vs_t": Vs_t, "Va_t": Va_t, "Vx_t": Vx_t,
        "n": n, "freq": freq, "Va_harm": Va_harm,
        "Gamma_S": (Zs_of_f(f0) - Z0) / (Zs_of_f(f0) + Z0),
        "Gamma_L": (ZL_of_f(f0) - Z0) / (ZL_of_f(f0) + Z0),
        "f0": f0,
    }
    if tap_divider is not None:
        Vout_harm = tap_divider(Va_harm, freq)
        out["Vout_harm"] = Vout_harm
        out["Vout_t"] = np.imag(Vout_harm[None, :] * np.exp(1j * np.outer(t, w))).sum(axis=1)
    return out
