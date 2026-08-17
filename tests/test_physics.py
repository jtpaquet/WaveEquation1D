"""Sanity checks for the FDTD line solver and its agreement with the
independent frequency-domain solver.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from waveline import lines, freqdomain as fd, scenarios as sc


def smoothed_step(t, rise=2e-9):
    """A step with a short raised-cosine edge (a true discontinuity would
    excite spurious numerical-dispersion ringing on any finite grid, which
    would contaminate these checks)."""
    if t <= 0:
        return 0.0
    if t >= rise:
        return 1.0
    return 0.5 - 0.5 * np.cos(np.pi * t / rise)


def test_matched_line_has_no_reflection():
    """Rs = Z0 = RL: a step should settle to Vs/2 with no overshoot/ringing."""
    src = lines.Source(smoothed_step, Rs=50.0, Ls=0.0)
    load = lines.ResistiveLoad(R=50.0)
    tl = lines.TLineFDTD(50.0, 0.66, 10.0, src, load, n_seg=200, cfl=0.9,
                          atten_db_per_m=0.0)
    res = tl.run(t_end=500e-9, record_every=5)
    va = res["Va"]
    assert abs(va[-1] - 0.5) < 0.02
    assert va.max() < 0.55


def test_open_load_gives_voltage_doubling():
    """Matched source + open load: step doubles at the load and the matched
    source absorbs the reflection with no further bounce."""
    src = lines.Source(smoothed_step, Rs=50.0, Ls=0.0)
    load = lines.ResistiveLoad(R=20e6)
    tl = lines.TLineFDTD(50.0, 0.66, 10.0, src, load, n_seg=200, cfl=0.9,
                          atten_db_per_m=0.0)
    res = tl.run(t_end=500e-9, record_every=5)
    assert abs(res["Va"][-1] - 1.0) < 0.02


def test_fdtd_settles_to_exact_frequency_domain_steady_state():
    """The transient FDTD solution and the independent phasor solver must
    agree on the final steady-state amplitude at the RC filter tap."""
    variant = "10k_1nF"
    src = sc.make_rogowski_source()
    load = sc.make_rc_load(variant)
    tl = lines.TLineFDTD(sc.Z0, sc.VF, sc.CABLE_LENGTH, src, load, n_seg=250,
                          cfl=0.9, atten_db_per_m=sc.atten_db_per_m(sc.F_SINE))
    t_end = 40 / sc.F_SINE
    res = tl.run(t_end=t_end, record_every=20)
    t, vout = res["t"], res["Vout"]
    mask = t > t[-1] - 1.0 / sc.F_SINE
    fdtd_amp = (vout[mask].max() - vout[mask].min()) / 2

    x = np.array([sc.CABLE_LENGTH])
    ss = fd.steady_state_sine(sc.F_SINE, sc.V_COIL_AMP, sc.coil_Zs_of_f,
                               sc.rc_ZL_of_f(variant), sc.Z0, sc.VF,
                               sc.CABLE_LENGTH, x,
                               atten_db_per_m_ref=sc.atten_db_per_m(sc.F_SINE),
                               f_ref=sc.F_SINE, tap_divider=sc.rc_tap_divider(variant))
    fd_amp = float(np.abs(ss["Vout_phasor"]))
    assert abs(fdtd_amp - fd_amp) / fd_amp < 0.05


def test_unterminated_square_wave_rings():
    """No series resistor: near-total reflection at both ends should produce
    ringing well beyond the incident step amplitude."""
    src = sc.make_square_source("none_0")
    load = sc.make_daq_load()
    tl = lines.TLineFDTD(sc.Z0, sc.VF, sc.CABLE_LENGTH, src, load, n_seg=150,
                          cfl=0.9, atten_db_per_m=sc.atten_db_per_m(sc.F_SQUARE))
    res = tl.run(t_end=2e-6, record_every=1)
    assert np.max(np.abs(res["Vout"])) > 0.3  # source amplitude is 0.25 V


def test_matched_series_resistor_suppresses_ringing():
    """50 ohm series resistor matches the line: after the initial doubled
    step at the open load, the received waveform should settle quickly with
    little residual ringing, unlike the unterminated case."""
    src = sc.make_square_source("matched_50")
    load = sc.make_daq_load()
    tl = lines.TLineFDTD(sc.Z0, sc.VF, sc.CABLE_LENGTH, src, load, n_seg=150,
                          cfl=0.9, atten_db_per_m=sc.atten_db_per_m(sc.F_SQUARE))
    res = tl.run(t_end=2e-6, record_every=1)
    t, vout = res["t"], res["Vout"]
    tail = vout[t > 1.5e-6]
    assert (tail.max() - tail.min()) < 0.01


if __name__ == "__main__":
    test_matched_line_has_no_reflection()
    test_open_load_gives_voltage_doubling()
    test_fdtd_settles_to_exact_frequency_domain_steady_state()
    test_unterminated_square_wave_rings()
    test_matched_series_resistor_suppresses_ringing()
    print("ALL SANITY TESTS PASSED")
