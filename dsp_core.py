"""
dsp_core.py  --  MAIN LOGIC (pure DSP, no GUI, no matplotlib, no sound card)
=============================================================================
Everything in here takes numbers/arrays in and returns numbers/arrays out,
so it can be unit-tested and reused by any front end (Tkinter, App Designer
port, Jupyter, command line ...).

Topics covered: sampling theorem, aliasing / frequency folding,
ideal (sinc) and zero-order-hold reconstruction, spectrum analysis,
anti-aliasing (ideal band-limiting) and audio synthesis for playback.
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

SIGNAL_TYPES = ("sine", "square", "triangle")
RECON_METHODS = ("sinc", "zoh")

AUDIO_FS = 44100                  # playback rate of the sound card (Hz)
MAX_DISPLAY_SAMPLES = 2000        # keeps the on-screen sampled plot light
MAX_SINC_ELEMENTS = 20_000_000    # guard for the sinc interpolation matrix


# --------------------------------------------------------------------------
# 1. Signal generation
# --------------------------------------------------------------------------
def _check_kind(kind):
    if kind not in SIGNAL_TYPES:
        raise ValueError(f"signal type must be one of {SIGNAL_TYPES}")


def waveform(kind, f, t, amp=1.0):
    """Ideal continuous-time waveform (square/triangle contain ALL harmonics)."""
    _check_kind(kind)
    s = np.sin(2 * np.pi * f * np.asarray(t, dtype=float))
    if kind == "sine":
        y = s
    elif kind == "square":
        y = np.where(s >= 0, 1.0, -1.0)
    else:  # triangle, same phase as the sine
        y = (2 / np.pi) * np.arcsin(s)
    return amp * y


def bandlimited_waveform(kind, f, t, f_max, amp=1.0):
    """
    Same waveform but with every harmonic at or above f_max removed.
    This is an *ideal anti-aliasing (brick-wall low-pass) filter* applied
    analytically through the Fourier series:
        square   = 4/pi    * sum_{k odd}  sin(k*phi) / k
        triangle = 8/pi^2  * sum_{k odd} (-1)^((k-1)/2) sin(k*phi) / k^2
    """
    _check_kind(kind)
    t = np.asarray(t, dtype=float)
    phi = 2 * np.pi * f * t
    if kind == "sine":
        return amp * np.sin(phi) if f < f_max else np.zeros_like(t)

    k_max = int(np.ceil(f_max / f)) - 1            # largest k with k*f < f_max
    ks = np.arange(1, k_max + 1, 2)                # odd harmonics only
    if ks.size == 0:
        return np.zeros_like(t)
    if kind == "square":
        w = (4 / np.pi) / ks
    else:
        w = (8 / np.pi ** 2) * (-1.0) ** ((ks - 1) // 2) / ks ** 2

    y = np.zeros_like(t)
    for i in range(0, ks.size, 200):               # chunked: low memory use
        k_blk = ks[i:i + 200]
        y += w[i:i + 200] @ np.sin(np.outer(k_blk, phi))
    return amp * y


# --------------------------------------------------------------------------
# 2. Aliasing maths
# --------------------------------------------------------------------------
def alias_frequency(f, fs):
    """Apparent frequency after sampling at fs: folded into [0, fs/2]."""
    f = np.asarray(f, dtype=float)
    return np.abs(f - fs * np.round(f / fs))


def harmonic_table(kind, f, fs, n_harm=8):
    """List of the first harmonics with their folded (aliased) frequencies."""
    _check_kind(kind)
    ks = [1] if kind == "sine" else list(range(1, 2 * n_harm, 2))
    rows = []
    for k in ks:
        if kind == "sine":
            amp = 1.0
        elif kind == "square":
            amp = 4 / (np.pi * k)
        else:
            amp = 8 / (np.pi ** 2 * k ** 2)
        freq = k * f
        rows.append({
            "k": k, "freq": freq, "amp": amp,
            "alias": float(alias_frequency(freq, fs)),
            "aliased": freq >= fs / 2,
        })
    return rows


# --------------------------------------------------------------------------
# 3. Sampling + reconstruction
# --------------------------------------------------------------------------
def reconstruct(t, x_s, fs, method="sinc"):
    """
    Rebuild a continuous-looking signal on time axis t from samples x_s.
      sinc : ideal Whittaker-Shannon interpolation  x(t)=sum x[n] sinc(fs t - n)
      zoh  : zero-order hold (staircase, what a real DAC outputs)
    Returns None if the sinc matrix would be too large.
    """
    if method not in RECON_METHODS:
        raise ValueError(f"method must be one of {RECON_METHODS}")
    if method == "zoh":
        idx = np.clip(np.floor(t * fs + 1e-9).astype(int), 0, x_s.size - 1)
        return x_s[idx]
    if t.size * x_s.size > MAX_SINC_ELEMENTS:
        return None
    n = np.arange(x_s.size)
    return np.sinc(t[:, None] * fs - n[None, :]) @ x_s


@dataclass
class SamplingResult:
    kind: str
    f: float
    fs: float
    nyquist: float
    f_alias: float
    aliased: bool
    antialias: bool
    t_dense: np.ndarray
    x_dense: np.ndarray       # original continuous signal
    n: np.ndarray             # sample indices
    t_s: np.ndarray           # sample instants
    x_s: np.ndarray           # sample values
    x_recon: np.ndarray       # reconstructed signal (None if too large)
    recon_method: str


def run_sampling(kind, f, fs, n_cycles=4, antialias=False, recon="sinc",
                 n_dense=4000, amp=1.0):
    """Sample a signal of frequency f at rate fs and reconstruct it."""
    _check_kind(kind)
    if f <= 0 or fs <= 0:
        raise ValueError("f and fs must be positive")

    duration = min(n_cycles / f, MAX_DISPLAY_SAMPLES / fs)
    duration = max(duration, 2.0 / fs)
    t_dense = np.linspace(0, duration, n_dense)
    x_dense = waveform(kind, f, t_dense, amp)

    n = np.arange(int(np.floor(duration * fs + 1e-9)) + 1)
    t_s = n / fs
    if antialias:
        x_s = bandlimited_waveform(kind, f, t_s, fs / 2, amp)
    else:
        x_s = waveform(kind, f, t_s, amp)

    x_recon = reconstruct(t_dense, x_s, fs, recon)
    f_alias = float(alias_frequency(f, fs))
    return SamplingResult(kind, f, fs, fs / 2, f_alias, f >= fs / 2, antialias,
                          t_dense, x_dense, n, t_s, x_s, x_recon, recon)


def summary_text(res):
    """One-line human readable verdict for the status bar."""
    if res.antialias and res.aliased:
        return (f"Anti-alias filter ON: components >= {res.nyquist:g} Hz removed "
                f"before sampling -> no aliasing (signal is attenuated).")
    if not res.aliased:
        if res.kind == "sine":
            return (f"fs = {res.fs:g} Hz > 2f = {2 * res.f:g} Hz : Nyquist "
                    f"criterion satisfied, perfect reconstruction.")
        bad = [r for r in harmonic_table(res.kind, res.f, res.fs, 200) if r["aliased"]]
        if not bad:
            return "Nyquist satisfied for all harmonics."
        return (f"Fundamental OK, but harmonic k={bad[0]['k']} ({bad[0]['freq']:g} Hz) "
                f"and above exceed {res.nyquist:g} Hz and fold back "
                f"(see harmonic table).")
    return (f"ALIASING: f = {res.f:g} Hz >= Nyquist {res.nyquist:g} Hz  ->  "
            f"appears as {res.f_alias:g} Hz.")


# --------------------------------------------------------------------------
# 4. Spectrum of the sampled signal
# --------------------------------------------------------------------------
def spectrum_of_sampled(kind, f, fs, antialias=False, n_fft=4096, amp=1.0):
    """Hann-windowed amplitude spectrum (0..fs/2) of N samples taken at fs."""
    t = np.arange(n_fft) / fs
    if antialias:
        x = bandlimited_waveform(kind, f, t, fs / 2, amp)
    else:
        x = waveform(kind, f, t, amp)
    w = np.hanning(n_fft)
    mag = 2 * np.abs(np.fft.rfft(x * w)) / np.sum(w)
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    return freqs, mag


def alias_map(fs, f_max, n=2000):
    """Input frequency vs apparent frequency (the 'folding' triangle wave)."""
    f_axis = np.linspace(0, f_max, n)
    return f_axis, alias_frequency(f_axis, fs)


# --------------------------------------------------------------------------
# 5. Audio synthesis (returns arrays only; playing them is audio_io's job)
# --------------------------------------------------------------------------
def _fade(y, ms=10, fs=AUDIO_FS):
    m = int(fs * ms / 1000)
    ramp = np.linspace(0, 1, m)
    y = y.copy()
    y[:m] *= ramp
    y[-m:] *= ramp[::-1]
    return y


def synth_original(kind, f, duration=1.0, fs_audio=AUDIO_FS, amp=0.4):
    """Clean reference tone at the sound-card rate."""
    t = np.arange(int(fs_audio * duration)) / fs_audio
    return _fade(bandlimited_waveform(kind, f, t, fs_audio / 2, amp), fs=fs_audio)


def synth_sampled(kind, f, fs, antialias=False, recon="sinc", duration=1.0,
                  fs_audio=AUDIO_FS, amp=0.4):
    """What you would hear after sampling at fs and reconstructing."""
    n = max(int(round(fs * duration)), 2)
    t = np.arange(n) / fs
    x = (bandlimited_waveform(kind, f, t, fs / 2, amp) if antialias
         else waveform(kind, f, t, amp))
    n_out = int(fs_audio * duration)
    if recon == "sinc":
        y = sps.resample(x, n_out)               # ideal band-limited interpolation
    else:
        t_a = np.arange(n_out) / fs_audio
        y = x[np.minimum((t_a * fs).astype(int), n - 1)]
    return _fade(y, fs=fs_audio)