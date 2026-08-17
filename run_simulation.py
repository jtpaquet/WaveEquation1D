#!/usr/bin/env python3
"""Generate transmission-line reflection/standing-wave visualizations for:

  Rogowski coil -> 10 cm lead -> 10 m RG58 -> RC low-pass filter
    rogowski_10k1nF   R=10k, C=1nF   (strongly mismatched load)
    rogowski_50_5uF   R=50,  C=5uF   (near impedance-matched load)

  250 kHz / 500 mVpp square wave op-amp output -> 10 m RG58 -> 10 cm lead -> 20 MOhm DAQ
    square_low_5      no *added* resistor (~5 ohm bare op-amp output)
    square_matched_50 50 ohm series resistor (source-side matched)
    square_partial_40 40 ohm series resistor (partial match)

Each scenario includes a short (10 cm) non-coax lead-wire stub at the coil
(source) end for the Rogowski scenarios, or at the DAQ (load) end for the
square-wave scenarios -- see waveline/scenarios.py's LEAD_* constants for the
assumed impedance/velocity of that stub. The impedance step where the lead
meets the 50 ohm coax is itself a (small) reflection point, marked in the
plots.

For each scenario this produces, in --outdir:
  <name>_reflections.gif    transient FDTD animation: watch the wave
                             propagate and bounce at both interfaces until
                             it settles.
  <name>_steady_loop.gif    the exact steady-state periodic solution,
                             looping (spatial profile + received waveform).
  <name>_steady.png         static steady-state comparison: received signal
                             vs. the original source (faded, twin axis) plus
                             a filmstrip of spatial snapshots across a period.

Usage:
  python run_simulation.py                      # all 5 scenarios, full outputs
  python run_simulation.py --scenario square_none_0
  python run_simulation.py --list
  python run_simulation.py --fast               # quicker/coarser, for previewing
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from waveline import lines, freqdomain as fd, scenarios as sc, plotting as pl


def _build_rogowski(variant):
    source = sc.make_rogowski_source()
    load = sc.make_rc_load(variant)
    Zs_of_f = sc.coil_Zs_of_f
    ZL_of_f = sc.rc_ZL_of_f(variant)
    tap = sc.rc_tap_divider(variant)
    label = sc.LPF_VARIANTS[variant]["label"]
    title = f"Rogowski coil signal through 10 cm lead + 10 m RG58 -> {label}"
    segments = [sc.lead_segment(), sc.cable_segment()]
    return dict(source=source, load=load, Zs_of_f=Zs_of_f, ZL_of_f=ZL_of_f,
                tap=tap, title=title, f0=sc.F_SINE, is_sine=True,
                vs_func=sc.coil_vs, vs_amp=sc.V_COIL_AMP, meas_name="Vout",
                meas_label="V_out (RC tap)", segments=segments)


def _build_square(variant):
    source = sc.make_square_source(variant)
    load = sc.make_daq_load()
    Zs_of_f = sc.square_Zs_of_f(variant)
    ZL_of_f = sc.daq_ZL_of_f
    label = sc.SERIES_R_VARIANTS[variant]["label"]
    title = f"500 mVpp 250 kHz square wave through 10 m RG58 + 10 cm lead -> 20 MΩ DAQ, {label}"
    segments = [sc.cable_segment(), sc.lead_segment()]
    return dict(source=source, load=load, Zs_of_f=Zs_of_f, ZL_of_f=ZL_of_f,
                tap=None, title=title, f0=sc.F_SQUARE, is_sine=False,
                vs_func=sc.square_vs, vs_amp=sc.SQUARE_AMP, meas_name="Va",
                meas_label="V_DAQ (after lead)", segments=segments)


SCENARIOS = {
    "rogowski_10k1nF": lambda: _build_rogowski("10k_1nF"),
    "rogowski_50_5uF": lambda: _build_rogowski("50_5uF"),
    "square_low_5": lambda: _build_square("low_5"),
    "square_matched_50": lambda: _build_square("matched_50"),
    "square_partial_40": lambda: _build_square("partial_40"),
}


def gamma_at_f0(cfg):
    f0 = cfg["f0"]
    Zs = cfg["Zs_of_f"](f0)
    ZL = cfg["ZL_of_f"](f0)
    Z0_source_side = cfg["segments"][0]["Z0"]
    Z0_load_side = cfg["segments"][-1]["Z0"]
    return ((Zs - Z0_source_side) / (Zs + Z0_source_side),
            (ZL - Z0_load_side) / (ZL + Z0_load_side))


def make_steady_state(cfg, n_points_x=161):
    total_length = sum(s["length"] for s in cfg["segments"])
    x = np.linspace(0, total_length, n_points_x)
    if cfg["is_sine"]:
        ss = fd.steady_state_sine(cfg["f0"], cfg["vs_amp"], cfg["Zs_of_f"], cfg["ZL_of_f"],
                                   None, None, None, x, tap_divider=cfg["tap"],
                                   segments=cfg["segments"])
        f0 = cfg["f0"]
        t_snap = np.linspace(0, 1.0 / f0, 6, endpoint=False)
        Vx_snap = np.imag(ss["V_x_phasor"][:, None] * np.exp(1j * 2 * np.pi * f0 * t_snap)[None, :]).T
        t_loop = np.linspace(0, 1.0 / f0, 90, endpoint=False)
        Vx_loop = np.imag(ss["V_x_phasor"][:, None] * np.exp(1j * 2 * np.pi * f0 * t_loop)[None, :])
        meas_t_loop = np.imag(ss["Vout_phasor"] * np.exp(1j * 2 * np.pi * f0 * t_loop))
        vout_key = "Vout_t"
    else:
        ss = fd.steady_state_square(cfg["f0"], cfg["vs_func"], cfg["Zs_of_f"], cfg["ZL_of_f"],
                                     None, None, None, x, n_max=161,
                                     tap_divider=cfg["tap"], n_time=1600,
                                     segments=cfg["segments"])
        f0 = cfg["f0"]
        t_full = ss["t"]
        idx6 = np.linspace(0, len(t_full) // 2 - 1, 6).astype(int)  # one period's worth
        t_snap = t_full[idx6]
        Vx_snap = ss["Vx_t"][:, idx6].T
        idx_loop = np.linspace(0, len(t_full) // 2 - 1, 120).astype(int)
        t_loop = t_full[idx_loop]
        Vx_loop = ss["Vx_t"][:, idx_loop]
        meas_t_loop = ss["Va_t"][idx_loop]
        vout_key = "Va_t"
    return x, ss, t_snap, Vx_snap, t_loop, Vx_loop, meas_t_loop, vout_key


def process_scenario(name, outdir, target_dx=0.02, n_periods=None, n_frames=200, fps=30,
                      skip_transient=False, skip_loop=False, skip_static=False):
    cfg = SCENARIOS[name]()
    print(f"[{name}] {cfg['title']}")
    gamma_s, gamma_l = gamma_at_f0(cfg)
    print(f"    Gamma_source={gamma_s:.3f} (|.|={abs(gamma_s):.3f})   "
          f"Gamma_load={gamma_l:.3f} (|.|={abs(gamma_l):.3f})")

    if n_periods is None:
        n_periods = 12 if cfg["is_sine"] else 8

    total_length = sum(s["length"] for s in cfg["segments"])
    n_seg = max(100, int(round(total_length / target_dx)))

    if not skip_transient:
        tl = lines.TLineFDTD(source=cfg["source"], load=cfg["load"], n_seg=n_seg,
                              cfl=0.9, segments=cfg["segments"])
        t_end = n_periods / cfg["f0"]
        n_steps = int(np.ceil(t_end / tl.dt))
        snapshot_every = max(1, n_steps // n_frames)
        record_every = max(1, n_steps // 4000)
        t0 = time.time()
        res = tl.run(t_end=t_end, record_every=record_every, snapshot_every=snapshot_every)
        print(f"    FDTD transient: {n_steps} steps, {len(res['snap_t'])} frames, "
              f"{time.time()-t0:.1f}s")
        path = os.path.join(outdir, f"{name}_reflections.gif")
        pl.animate_propagation(tl, res, cfg["meas_name"], cfg["meas_label"],
                                cfg["title"], gamma_s, gamma_l, path, fps=fps,
                                n_frames=n_frames, f0=cfg["f0"], segments=cfg["segments"])
        print(f"    wrote {path}")

    x, ss, t_snap, Vx_snap, t_loop, Vx_loop, meas_t_loop, vout_key = make_steady_state(cfg)

    if not skip_loop:
        path = os.path.join(outdir, f"{name}_steady_loop.gif")
        pl.animate_steady_state_loop(x, Vx_loop, t_loop, meas_t_loop, cfg["title"],
                                      gamma_s, gamma_l, path, fps=fps,
                                      meas_label=cfg["meas_label"], f0=cfg["f0"],
                                      segments=cfg["segments"])
        print(f"    wrote {path}")

    if not skip_static:
        f0 = cfg["f0"]
        if cfg["is_sine"]:
            t_cmp = ss["t"]
            vs_cmp = ss["Vs_t"]
            vout_cmp = ss["Vout_t"]
        else:
            n_half = len(ss["t"]) // 2
            t_cmp = ss["t"][:n_half]
            vs_cmp = ss["Vs_t"][:n_half]
            vout_cmp = ss[vout_key][:n_half]
        path = os.path.join(outdir, f"{name}_steady.png")
        pl.plot_steady_state_comparison(t_cmp, vs_cmp, vout_cmp, x, Vx_snap, t_snap,
                                         cfg["title"], gamma_s, gamma_l, path,
                                         meas_label=cfg["meas_label"], f0=f0,
                                         segments=cfg["segments"])
        print(f"    wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=list(SCENARIOS), default=None,
                     help="run a single scenario (default: all)")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fast", action="store_true",
                     help="coarser grid / fewer frames, for a quick preview")
    ap.add_argument("--skip-transient", action="store_true")
    ap.add_argument("--skip-loop", action="store_true")
    ap.add_argument("--skip-static", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, builder in SCENARIOS.items():
            print(f"{k}: {builder()['title']}")
        return

    os.makedirs(args.outdir, exist_ok=True)
    names = [args.scenario] if args.scenario else list(SCENARIOS)

    # target_dx is chosen so the 10 cm lead stub still gets a handful of
    # grid cells (not just the main 10 m cable's resolution)
    target_dx = 0.05 if args.fast else 0.02
    n_frames = 100 if args.fast else 200
    fps = 24 if args.fast else 30

    for name in names:
        process_scenario(name, args.outdir, target_dx=target_dx, n_frames=n_frames, fps=fps,
                          skip_transient=args.skip_transient,
                          skip_loop=args.skip_loop,
                          skip_static=args.skip_static)


if __name__ == "__main__":
    main()
