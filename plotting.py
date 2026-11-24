"""
plotting.py  --  VISUALISATION LOGIC (matplotlib only)
=======================================================
Knows how to draw a SamplingResult onto matplotlib axes.
It does not know about Tkinter, buttons or sound, and does not do any DSP
itself: all numbers come from dsp_core.
"""
import numpy as np
from matplotlib.figure import Figure

import dsp_core

C_ORIG, C_SAMP, C_RECON, C_ALIAS = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"


def make_figure():
    """Create the empty figure with its three axes."""
    fig = Figure(figsize=(9, 7), dpi=100, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    ax_time = fig.add_subplot(gs[0, :])
    ax_spec = fig.add_subplot(gs[1, 0])
    ax_map = fig.add_subplot(gs[1, 1])
    return fig, (ax_time, ax_spec, ax_map)


def draw_time(ax, res, show_recon=True):
    ax.clear()
    ax.plot(res.t_dense * 1e3, res.x_dense, color=C_ORIG, lw=1.4,
            label="Original (continuous)")
    markerline, stemlines, _ = ax.stem(res.t_s * 1e3, res.x_s, basefmt=" ",
                                       label=f"Samples (fs = {res.fs:g} Hz)")
    markerline.set_color(C_SAMP)
    stemlines.set_color(C_SAMP)
    stemlines.set_linewidth(0.8)
    if show_recon and res.x_recon is not None:
        name = "Sinc reconstruction" if res.recon_method == "sinc" else "Zero-order hold"
        ax.plot(res.t_dense * 1e3, res.x_recon, color=C_RECON, lw=1.4, ls="--",
                label=name)
    elif show_recon:
        ax.text(0.5, 0.9, "(sinc reconstruction skipped: too many samples)",
                transform=ax.transAxes, ha="center", fontsize=8)
    ax.set_title("Time domain: original vs sampled vs reconstructed")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.4, 1.4)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=3)


def draw_spectrum(ax, freqs, mag, res):
    ax.clear()
    ax.plot(freqs, mag, color=C_SAMP, lw=1.2)
    ax.fill_between(freqs, mag, color=C_SAMP, alpha=0.15)
    if not res.aliased:
        ax.axvline(res.f, color=C_ORIG, ls=":", label=f"True f = {res.f:g} Hz")
    else:
        ax.axvline(res.f_alias, color=C_ALIAS, ls="--",
                   label=f"Alias = {res.f_alias:g} Hz\n(true f = {res.f:g} Hz)")
    ax.set_xlim(0, res.nyquist)
    ax.set_title(f"Spectrum of sampled signal (0 to Nyquist = {res.nyquist:g} Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def draw_alias_map(ax, res):
    ax.clear()
    f_max = max(1.2 * res.f, 2.5 * res.fs)
    f_axis, f_app = dsp_core.alias_map(res.fs, f_max)
    ax.plot(f_axis, f_app, color="gray", lw=1.2, label="Apparent frequency")
    ax.axvline(res.nyquist, color="k", ls=":", lw=0.8)
    ax.axvline(res.fs, color="k", ls=":", lw=0.8)
    ax.plot([res.f], [res.f_alias], "o", ms=9,
            color=C_ALIAS if res.aliased else C_ORIG, label="Current signal")
    ax.set_title("Frequency folding map")
    ax.set_xlabel("True frequency (Hz)")
    ax.set_ylabel("Apparent frequency (Hz)")
    ax.set_ylim(0, res.nyquist * 1.1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)


def render(axes, res, freqs, mag, show_recon=True):
    """Redraw all three plots."""
    ax_time, ax_spec, ax_map = axes
    draw_time(ax_time, res, show_recon)
    draw_spectrum(ax_spec, freqs, mag, res)
    draw_alias_map(ax_map, res)


def harmonic_text(rows, fs):
    """Plain-text table of harmonics (shown in a text box by the GUI)."""
    lines = [f"{'k':>3} {'freq(Hz)':>10} {'amp':>6} {'appears at':>11}"]
    for r in rows:
        flag = "  <-- folded" if r["aliased"] else ""
        lines.append(f"{r['k']:>3} {r['freq']:>10.1f} {r['amp']:>6.3f} "
                     f"{r['alias']:>11.1f}{flag}")
    return "\n".join(lines)