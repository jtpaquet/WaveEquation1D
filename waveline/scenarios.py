"""Parameter sets for the physical setups the user described, and small
helpers that build both the time-domain (FDTD) objects and the matching
frequency-domain impedance functions from the *same* numbers, so the two
solvers never disagree about what circuit they're simulating.
"""
from __future__ import annotations

import math

import numpy as np

from . import lines

# ---------------------------------------------------------------- cable ----
Z0 = 50.0
VF = 0.66                      # RG58 velocity factor (~2/3 c)
CABLE_LENGTH = 10.0             # m
# RG58 attenuation figure used for damping: ~4.3 dB/100 m @ 1 MHz, scaled by
# sqrt(f) below f_ref for the frequency-domain solver, and evaluated once at
# 250 kHz for the (frequency-independent) FDTD series resistance.
ATTEN_DB_PER_100M_AT_1MHZ = 4.3
ATTEN_F_REF = 1.0e6


def atten_db_per_m(f):
    return (ATTEN_DB_PER_100M_AT_1MHZ / 100.0) * np.sqrt(np.abs(f) / ATTEN_F_REF)


# ------------------------------------------------------------- Rogowski ----
F_SINE = 250e3
COIL_TURNS = 80
COIL_AREA_CM2 = 1.0
COIL_R = 2.0            # ohm, coil DC resistance
COIL_L = 23.6e-6        # H, coil self-inductance (source series impedance)
COIL_SENSITIVITY = 0.200  # V/A, *calibrated at 250 kHz* (as specified)
I_WIRE_AMP = 30.0        # A
V_COIL_AMP = COIL_SENSITIVITY * I_WIRE_AMP  # 6 V EMF amplitude


def coil_vs(t):
    return V_COIL_AMP * np.sin(2 * np.pi * F_SINE * t)


def make_rogowski_source(dt_hint=None):
    return lines.Source(coil_vs, Rs=COIL_R, Ls=COIL_L)


def coil_Zs_of_f(f):
    return COIL_R + 1j * 2 * np.pi * f * COIL_L


LPF_VARIANTS = {
    "10k_1nF": dict(R=10e3, C=1e-9,
                    label="10 kΩ / 1 nF LPF (mismatched, near-open)"),
    "50_5uF": dict(R=50.0, C=5e-6,
                   label="50 Ω / 5 µF LPF (near impedance-matched)"),
}


def make_rc_load(variant):
    p = LPF_VARIANTS[variant]
    return lines.SeriesRCLoad(R=p["R"], C=p["C"])


def rc_ZL_of_f(variant):
    p = LPF_VARIANTS[variant]
    R, C = p["R"], p["C"]

    def ZL(f):
        return R + 1.0 / (1j * 2 * np.pi * f * C)
    return ZL


def rc_tap_divider(variant):
    p = LPF_VARIANTS[variant]
    R, C = p["R"], p["C"]

    def tap(Va, f):
        Zc = 1.0 / (1j * 2 * np.pi * f * C)
        return Va * Zc / (R + Zc)
    return tap


# ----------------------------------------------------------- square wave ---
F_SQUARE = 250e3
SQUARE_VPP = 0.5                 # "500 mV square wave" taken as 500 mV pk-pk
SQUARE_AMP = SQUARE_VPP / 2.0     # +/- amplitude, bipolar about 0 V
R_DAQ = 20e6                      # ohm, DAQ high-Z input
# Real op-amps don't produce mathematically instantaneous edges; a fast
# general-purpose op-amp driving 0.5 Vpp has a rise time on the order of
# tens of ns. This also keeps the FDTD excitation smooth on the simulation
# grid (a true discontinuity would excite spurious numerical-dispersion
# ringing unrelated to real cable reflections). Still >100x faster than
# the 4 us period, so the waveform looks like a clean square wave.
SQUARE_RISE_TIME = 20e-9

SERIES_R_VARIANTS = {
    "matched_50": dict(Rs=50.0, label="50 Ω series resistor (matched)"),
    "none_0": dict(Rs=0.0, label="no series resistor (0 Ω, ideal op-amp out)"),
    "partial_40": dict(Rs=40.0, label="40 Ω series resistor (partial match)"),
}


def _make_smoothed_square(f0, amplitude, rise_time):
    """Bipolar square wave with raised-cosine (smoothstep) edges of the
    given rise_time, period 1/f0. Returns a scalar-in/scalar-out function.
    """
    T = 1.0 / f0
    half = T / 2.0
    half_rt = rise_time / 2.0

    def vs(t):
        tp = t % T
        d_rise = min(tp, T - tp)
        if d_rise < half_rt:
            x = tp if tp <= half else tp - T
            xc = max(-1.0, min(1.0, x / half_rt))
            return amplitude * math.sin(xc * math.pi / 2.0)
        d_fall = abs(tp - half)
        if d_fall < half_rt:
            x = tp - half
            xc = max(-1.0, min(1.0, x / half_rt))
            return -amplitude * math.sin(xc * math.pi / 2.0)
        return amplitude if tp < half else -amplitude

    return vs


square_vs = _make_smoothed_square(F_SQUARE, SQUARE_AMP, SQUARE_RISE_TIME)


def make_square_source(variant):
    Rs = SERIES_R_VARIANTS[variant]["Rs"]
    return lines.Source(square_vs, Rs=Rs, Ls=0.0)


def square_Zs_of_f(variant):
    Rs = SERIES_R_VARIANTS[variant]["Rs"]

    def Zs(f):
        return np.full_like(np.asarray(f, dtype=float), Rs, dtype=complex) \
            if np.ndim(f) else complex(Rs)
    return Zs


def make_daq_load():
    return lines.ResistiveLoad(R=R_DAQ)


def daq_ZL_of_f(f):
    return np.full_like(np.asarray(f, dtype=float), R_DAQ, dtype=complex) \
        if np.ndim(f) else complex(R_DAQ)
