"""Matplotlib visualizations: the transient reflection animation and the
steady-state comparison figure (measured signal vs. the original, faded).
"""
from __future__ import annotations

import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SOURCE_COLOR = "#d62728"
LOAD_COLOR = "#1f77b4"
LINE_COLOR = "#2b2b2b"
FADED_COLOR = "#9a9a9a"

C_LIGHT = 299_792_458.0


def _fmt_gamma(g):
    return f"{g.real:+.2f}{g.imag:+.2f}j  (|Γ|={abs(g):.2f})"


def _wrap_title(s, width=62):
    return "\n".join(textwrap.wrap(s, width=width))


def _elec_length_note(f0, segments, width=78):
    """segments: list of dicts with 'length' (m), 'Z0' (ohm), 'vf'."""
    total_deg = 0.0
    parts = []
    for s in segments:
        v = s["vf"] * C_LIGHT
        deg = 360.0 * f0 * s["length"] / v
        total_deg += deg
        length_str = f"{s['length']*100:.0f} cm" if s["length"] < 1 else f"{s['length']:.0f} m"
        parts.append(f"{length_str} @ Z0={s['Z0']:.0f}Ω")
    breakdown = " + ".join(parts)
    if total_deg < 30:
        s = (f"electrical length at {f0/1e3:.0f} kHz: {total_deg:.1f}° total ({breakdown}) "
             f"— electrically short, so the spatial profile barely varies; the reflections "
             f"show up in TIME instead")
    else:
        s = f"electrical length at {f0/1e3:.0f} kHz: {total_deg:.0f}° total ({breakdown})"
    return "\n".join(textwrap.wrap(s, width=width))


def _draw_junctions(ax, junctions, y_inside):
    """Mark internal segment (Z0 step) boundaries with a dotted line and a
    label placed just inside the top of the axes, clear of the source/load
    Gamma annotations which sit just above the axes."""
    for xj in junctions:
        ax.axvline(xj, color="0.55", ls=":", lw=1.1)
        ax.text(xj, y_inside, "Z0 step", color="0.45", ha="center", va="top",
                fontsize=7, style="italic")


def animate_propagation(tl, run_result, meas_name, meas_label, title,
                         gamma_s, gamma_l, save_path, fps=30, n_frames=240,
                         window_periods=None, f0=None, segments=None):
    """Animate V(x,t) (top) and the measured signal building up over time
    (bottom) from the recorded FDTD snapshots.
    """
    snap_t = run_result["snap_t"]
    snap_V = run_result["snap_V"]
    t_full = run_result["t"]
    v_full = run_result[meas_name]

    n_snap = len(snap_t)
    idx = np.unique(np.linspace(0, n_snap - 1, min(n_frames, n_snap)).astype(int))

    x = tl.x
    vmax = max(np.max(np.abs(snap_V)), 1e-9) * 1.15

    fig, (ax_x, ax_t) = plt.subplots(2, 1, figsize=(10, 7.5),
                                      gridspec_kw={"height_ratios": [1.1, 1]})
    fig.suptitle(_wrap_title(title), fontsize=11, fontweight="bold")
    if f0 is not None and segments is not None:
        ax_x.set_title(_elec_length_note(f0, segments), fontsize=8,
                        color="0.4", style="italic")

    line_x, = ax_x.plot([], [], color=LINE_COLOR, lw=1.8)
    ax_x.axvline(0, color=SOURCE_COLOR, ls="--", lw=1)
    ax_x.axvline(x[-1], color=LOAD_COLOR, ls="--", lw=1)
    if getattr(tl, "junctions", None):
        _draw_junctions(ax_x, tl.junctions, vmax * 0.95)
    ax_x.set_xlim(x[0], x[-1])
    ax_x.set_ylim(-vmax, vmax)
    ax_x.set_xlabel("position along cable x [m]")
    ax_x.set_ylabel("V(x, t) [V]")
    ax_x.text(0, vmax * 1.02, f"source\nΓ={_fmt_gamma(gamma_s)}", color=SOURCE_COLOR,
              ha="left", va="bottom", fontsize=8)
    ax_x.text(x[-1], vmax * 1.02, f"load\nΓ={_fmt_gamma(gamma_l)}", color=LOAD_COLOR,
              ha="right", va="bottom", fontsize=8)
    time_txt = ax_x.text(0.02, 0.05, "", transform=ax_x.transAxes, fontsize=9,
                          family="monospace")

    tmax = snap_t[idx[-1]]
    mask_t = t_full <= tmax + 1e-15
    ax_t.plot(t_full[mask_t] * 1e9, v_full[mask_t], color=FADED_COLOR, lw=0.8, alpha=0.35)
    line_t, = ax_t.plot([], [], color=LOAD_COLOR, lw=1.6)
    cursor = ax_t.axvline(0, color="k", lw=0.8, alpha=0.6)
    vt_max = max(np.max(np.abs(v_full[mask_t])), 1e-9) * 1.15
    ax_t.set_xlim(0, tmax * 1e9)
    ax_t.set_ylim(-vt_max, vt_max)
    ax_t.set_xlabel("time [ns]")
    ax_t.set_ylabel(f"{meas_label} [V]")
    ax_t.grid(alpha=0.25)
    ax_x.grid(alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.90 if f0 is not None else 0.93])

    def init():
        line_x.set_data([], [])
        line_t.set_data([], [])
        return line_x, line_t, time_txt, cursor

    def update(frame_i):
        k = idx[frame_i]
        line_x.set_data(x, snap_V[k])
        t_now = snap_t[k]
        time_txt.set_text(f"t = {t_now*1e9:7.1f} ns")
        m = t_full <= t_now
        line_t.set_data(t_full[m] * 1e9, v_full[m])
        cursor.set_xdata([t_now * 1e9, t_now * 1e9])
        return line_x, line_t, time_txt, cursor

    anim = animation.FuncAnimation(fig, update, frames=len(idx), init_func=init,
                                    blit=False, interval=1000 / fps)
    anim.save(save_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return save_path


def animate_steady_state_loop(x, Vx_t, t, meas_t, title, gamma_s, gamma_l,
                               save_path, fps=30, meas_label="V_out", f0=None,
                               segments=None):
    """Loop the exact periodic steady state: spatial profile (top) and the
    measured waveform with a moving phase cursor (bottom).
    """
    n_t = len(t)
    vmax = max(np.max(np.abs(Vx_t)), 1e-9) * 1.15
    vt_max = max(np.max(np.abs(meas_t)), 1e-9) * 1.15

    fig, (ax_x, ax_t) = plt.subplots(2, 1, figsize=(10, 7.5),
                                      gridspec_kw={"height_ratios": [1.1, 1]})
    fig.suptitle(_wrap_title(title + "  (steady state, looping)"), fontsize=11,
                 fontweight="bold")
    if f0 is not None and segments is not None:
        ax_x.set_title(_elec_length_note(f0, segments), fontsize=8,
                        color="0.4", style="italic")

    line_x, = ax_x.plot([], [], color=LINE_COLOR, lw=1.8)
    ax_x.axvline(0, color=SOURCE_COLOR, ls="--", lw=1)
    ax_x.axvline(x[-1], color=LOAD_COLOR, ls="--", lw=1)
    if segments is not None and len(segments) > 1:
        junctions = np.cumsum([s["length"] for s in segments])[:-1]
        _draw_junctions(ax_x, junctions, vmax * 0.95)
    ax_x.set_xlim(x[0], x[-1])
    ax_x.set_ylim(-vmax, vmax)
    ax_x.set_xlabel("position along cable x [m]")
    ax_x.set_ylabel("V(x, t) [V]  (steady state)")
    ax_x.text(0, vmax * 1.02, f"source Γ={_fmt_gamma(gamma_s)}", color=SOURCE_COLOR,
              ha="left", va="bottom", fontsize=8)
    ax_x.text(x[-1], vmax * 1.02, f"load Γ={_fmt_gamma(gamma_l)}", color=LOAD_COLOR,
              ha="right", va="bottom", fontsize=8)
    ax_x.grid(alpha=0.25)

    ax_t.plot(t * 1e9, meas_t, color=LOAD_COLOR, lw=1.4)
    cursor = ax_t.axvline(0, color="k", lw=1.0)
    ax_t.set_xlim(t[0] * 1e9, t[-1] * 1e9)
    ax_t.set_ylim(-vt_max, vt_max)
    ax_t.set_xlabel("time [ns]")
    ax_t.set_ylabel(f"{meas_label} [V]")
    ax_t.grid(alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.90 if f0 is not None else 0.93])

    def init():
        line_x.set_data([], [])
        return line_x, cursor

    def update(k):
        line_x.set_data(x, Vx_t[:, k] if Vx_t.ndim == 2 else Vx_t[k])
        cursor.set_xdata([t[k] * 1e9, t[k] * 1e9])
        return line_x, cursor

    anim = animation.FuncAnimation(fig, update, frames=n_t, init_func=init,
                                    blit=False, interval=1000 / fps)
    anim.save(save_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return save_path


def plot_steady_state_comparison(t, vs_t, vout_t, x, Vx_snapshots, snapshot_times,
                                  title, gamma_s, gamma_l, save_path,
                                  meas_label="V_out", t_unit_ns=True, f0=None,
                                  segments=None):
    """Static figure: (top) measured output vs. source, source faded on a
    twin axis so both can be read at their own natural scale while time
    (and hence phase) stays directly comparable; (bottom) a filmstrip of
    spatial snapshots across one period.
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9),
                                          gridspec_kw={"height_ratios": [1, 1]})
    fig.suptitle(_wrap_title(title + "  —  steady state"), fontsize=11, fontweight="bold")

    tt = t * 1e9 if t_unit_ns else t
    ax_top.plot(tt, vout_t, color=LOAD_COLOR, lw=2.0, label=meas_label, zorder=3)
    ax_top.set_xlabel("time [ns]" if t_unit_ns else "time [s]")
    ax_top.set_ylabel(f"{meas_label} [V]", color=LOAD_COLOR)
    ax_top.tick_params(axis="y", labelcolor=LOAD_COLOR)
    ax_top.grid(alpha=0.25)

    ax_src = ax_top.twinx()
    ax_src.plot(tt, vs_t, color=FADED_COLOR, lw=2.4, alpha=0.45,
                label="original source signal", zorder=1)
    ax_src.set_ylabel("source [V]", color=FADED_COLOR)
    ax_src.tick_params(axis="y", labelcolor=FADED_COLOR)

    lines1, labels1 = ax_top.get_legend_handles_labels()
    lines2, labels2 = ax_src.get_legend_handles_labels()
    ax_top.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax_top.text(0.01, 0.02,
                f"Γ_source={_fmt_gamma(gamma_s)}\nΓ_load={_fmt_gamma(gamma_l)}",
                transform=ax_top.transAxes, fontsize=8, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    cmap = plt.get_cmap("viridis")
    n = len(snapshot_times)
    for i, (tv, Vx) in enumerate(zip(snapshot_times, Vx_snapshots)):
        ax_bot.plot(x, Vx, color=cmap(i / max(n - 1, 1)), lw=1.3,
                    label=f"t={tv*1e9:.0f} ns" if t_unit_ns else f"t={tv:.2e}s")
    ax_bot.axvline(0, color=SOURCE_COLOR, ls="--", lw=1)
    ax_bot.axvline(x[-1], color=LOAD_COLOR, ls="--", lw=1)
    if segments is not None and len(segments) > 1:
        junctions = np.cumsum([s["length"] for s in segments])[:-1]
        ymax_bot = max(np.max(np.abs(Vx_snapshots)), 1e-9)
        _draw_junctions(ax_bot, junctions, ymax_bot * 0.95)
    ax_bot.set_xlabel("position along cable x [m]")
    ax_bot.set_ylabel("V(x) [V]")
    bot_title = "spatial profile at several instants across one period"
    if f0 is not None and segments is not None:
        bot_title += "\n" + _elec_length_note(f0, segments)
    ax_bot.set_title(bot_title, fontsize=9, color="0.35")
    ax_bot.legend(fontsize=7, ncol=3, loc="upper right")
    ax_bot.grid(alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.subplots_adjust(hspace=0.55)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path
