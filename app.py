"""
app.py  --  USER INTERFACE (Streamlit)
======================================
Wires browser controls to the existing DSP & plotting modules:
    dsp_core  -> computes
    plotting  -> draws
"""
import io
import wave
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import dsp_core
import plotting

# ------------------------------------------------------------------ Page Config
st.set_page_config(
    page_title="Sampling & Aliasing Visualizer | CS-419 DSP",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ Styling
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
        header[data-testid="stHeader"] { background: transparent; }

        /* Hero */
        .hero {
            padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 1rem;
            background: linear-gradient(120deg, #14213d 0%, #1f4068 100%);
            color: #fff;
        }
        .hero h1 { margin: 0; font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; }
        .hero p  { margin: .25rem 0 0; opacity: .8; font-size: .95rem; }

        /* Status banner */
        .banner {
            display: flex; gap: .8rem; align-items: flex-start;
            padding: .9rem 1.1rem; border-radius: 12px; margin-bottom: 1rem;
            border-left: 6px solid; font-size: .95rem; line-height: 1.45;
        }
        .banner.ok  { background: #e8f6ee; border-color: #1e9e5a; color: #14532d; }
        .banner.bad { background: #fdecea; border-color: #d93025; color: #7a1c14; }
        .banner.warn { background: #fff4e0; border-color: #f0a202; color: #6b4600; }
        .banner.info { background: #e7f0fb; border-color: #2f6fd6; color: #14346b; }
        .banner .icon { font-size: 1.4rem; line-height: 1.2; }
        .banner b { display: block; margin-bottom: .1rem; font-size: 1rem; }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: rgba(120,130,150,.08); border: 1px solid rgba(120,130,150,.22);
            padding: .7rem .9rem; border-radius: 12px;
        }
        div[data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 600; }
        div[data-testid="stMetricLabel"] p { font-size: .8rem; opacity: .75; }

        /* Tabs */
        button[data-baseweb="tab"] { font-weight: 600; padding: .6rem 1rem; }

        /* Sidebar */
        section[data-testid="stSidebar"] .stButton > button { border-radius: 10px; text-align: left; }
        section[data-testid="stSidebar"] h3 { margin-bottom: .2rem; }

        /* Hide Streamlit footer */
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

PRESETS = {
    "✅ Good sampling (100 Hz @ 1 kHz)": ("sine", 100, 1000),
    "🟡 Just above Nyquist (450 Hz @ 1 kHz)": ("sine", 450, 1000),
    "❌ Aliased (900 → 100 Hz)": ("sine", 900, 1000),
    "🔲 Square, harmonics fold (300 Hz @ 2 kHz)": ("square", 300, 2000),
}

DEFAULTS = dict(
    kind="sine", f=100, fs=1000, cycles=4,
    antialias=False, recon="sinc", show_recon=True,
)

# ------------------------------------------------------------------ Session State
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(kind: str, f: int, fs: int):
    st.session_state.kind = kind
    st.session_state.f = f
    st.session_state.fs = fs


def reset_all():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def set_safe_fs():
    """Set fs to 2.5x f (safely above Nyquist), clipped to slider range."""
    st.session_state.fs = int(min(20000, max(2, round(st.session_state.f * 2.5))))


def array_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Converts a float/int numpy audio array into in-memory WAV bytes for Streamlit."""
    arr = np.asarray(samples, dtype=np.float32)
    peak = np.max(np.abs(arr)) if arr.size else 0
    if peak > 0:
        arr = arr / peak  # normalize safely to [-1, 1]
    pcm16 = (arr * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def cached_original_wav(kind, f):
    return array_to_wav_bytes(dsp_core.synth_original(kind, f), dsp_core.AUDIO_FS)


@st.cache_data(show_spinner=False)
def cached_sampled_wav(kind, f, fs, antialias, recon):
    return array_to_wav_bytes(
        dsp_core.synth_sampled(kind, f, fs, antialias, recon), dsp_core.AUDIO_FS
    )


# ------------------------------------------------------------------ Sidebar Controls
with st.sidebar:
    st.title("🎛️ DSP Controls")
    st.caption("CS-419 Digital Signal Processing")

    st.subheader("⚡ Quick presets")
    current = (st.session_state.kind, st.session_state.f, st.session_state.fs)
    for label, vals in PRESETS.items():
        st.button(
            label,
            use_container_width=True,
            on_click=apply_preset,
            args=vals,
            type="primary" if vals == current else "secondary",
        )

    st.divider()

    st.subheader("📡 Signal & sampling")
    st.selectbox(
        "Signal type",
        options=dsp_core.SIGNAL_TYPES,
        key="kind",
        help="Non-sine signals contain harmonics that can fold back (alias) too.",
    )
    st.slider(
        "Signal frequency f (Hz)", min_value=1, max_value=5000, key="f",
        help="Fundamental frequency of the continuous signal.",
    )
    st.slider(
        "Sampling frequency fs (Hz)", min_value=2, max_value=20000, key="fs",
        help="How many samples per second are taken. Needs fs > 2f to avoid aliasing.",
    )
    st.button(
        "Set safe fs (2.5 × f)", use_container_width=True, on_click=set_safe_fs,
        help="Jump to a sampling rate comfortably above Nyquist.",
    )
    st.slider(
        "Cycles shown", min_value=1, max_value=10, key="cycles",
        help="How many periods of the signal appear in the time plot.",
    )

    st.divider()

    st.subheader("🔧 Filter & reconstruction")
    st.toggle(
        "Anti-aliasing filter (ideal LPF at fs/2)", key="antialias",
        help="Removes everything above fs/2 before sampling.",
    )
    st.toggle("Show reconstructed curve", key="show_recon")
    st.radio(
        "Reconstruction method",
        options=["sinc", "zoh"],
        format_func=lambda x: "Ideal sinc interpolation" if x == "sinc" else "Zero-order hold (ZOH)",
        key="recon",
        disabled=not st.session_state.show_recon,
    )

    st.divider()
    st.button("↺ Reset all", use_container_width=True, on_click=reset_all)

# Read values from state
kind = st.session_state.kind
f = st.session_state.f
fs = st.session_state.fs
cycles = st.session_state.cycles
antialias = st.session_state.antialias
recon = st.session_state.recon
show_recon = st.session_state.show_recon

# ------------------------------------------------------------------ DSP Computation
try:
    res = dsp_core.run_sampling(kind, f, fs, cycles, antialias, recon)
    freqs, mag = dsp_core.spectrum_of_sampled(kind, f, fs, antialias)
except Exception as e:  # keep the UI alive on bad parameter combos
    st.error(f"Could not compute this configuration: {e}")
    st.stop()

nyquist = fs / 2.0
# Apparent frequency of the fundamental after sampling
alias_f = res.f_alias

# Harmonics that land above Nyquist (only relevant when no anti-alias filter)
harm_rows = dsp_core.harmonic_table(kind, f, fs, n_harm=200 if kind != "sine" else 8)
folding = [r for r in harm_rows if r["aliased"]]

# ------------------------------------------------------------------ Header
st.markdown(
    """
    <div class="hero">
        <h1>Sampling & Aliasing Visualizer</h1>
        <p>Change the signal and sampling rate, then see and hear what happens in time, frequency, and sound.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if antialias:
    state, title = "info", "Anti-aliasing filter on"
    if res.aliased:
        msg = (f"Everything at or above {nyquist:g} Hz is removed before sampling, "
               f"so nothing folds back. The {f:g} Hz signal is blocked, so the output is silent.")
    elif kind == "sine":
        msg = f"fs = {fs:g} Hz > 2f = {2 * f:g} Hz. The filter has no effect on this signal."
    else:
        msg = (f"Harmonics at or above {nyquist:g} Hz are removed before sampling, "
               f"so nothing folds back. The wave looks smoother because it lost those harmonics.")
elif res.aliased:
    state, title = "bad", "Aliasing detected"
    msg = (f"f = {f:g} Hz is at or above the Nyquist limit ({nyquist:g} Hz). "
           f"It appears as {res.f_alias:g} Hz after sampling.")
elif folding:
    state, title = "warn", "Harmonics are aliasing"
    first = folding[0]
    msg = (f"The fundamental ({f:g} Hz) is fine, but harmonic k={first['k']} "
           f"({first['freq']:g} Hz) and higher are above {nyquist:g} Hz and fold back. "
           f"See the Harmonics tab.")
else:
    state, title = "ok", "Valid sampling"
    msg = (f"fs = {fs:g} Hz > 2f = {2 * f:g} Hz. Nyquist criterion is satisfied."
           if kind == "sine" else "Nyquist criterion is satisfied for all harmonics.")

ICONS = {"ok": "✅", "warn": "🟠", "bad": "⚠️", "info": "🛡️"}
st.markdown(
    f'<div class="banner {state}"><span class="icon">{ICONS[state]}</span>'
    f"<div><b>{title}</b>{msg}</div></div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ Metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Signal freq (f)", f"{f:,} Hz")
m2.metric("Sampling rate (fs)", f"{fs:,} Hz")
m3.metric("Nyquist limit (fs/2)", f"{nyquist:,.1f} Hz")
m4.metric(
    "Ratio (fs / f)",
    f"{fs / f:.2f}×",
    delta="Above Nyquist" if fs > 2 * f else "Below Nyquist",
    delta_color="normal" if fs > 2 * f else "inverse",
)
m5.metric(
    "Apparent freq",
    f"{alias_f:,.1f} Hz",
    delta=None if alias_f == f else f"{alias_f - f:+,.1f} Hz shift",
    delta_color="off",
    help="Frequency you would measure for the fundamental after sampling.",
)

st.write("")

# ------------------------------------------------------------------ Tabs
tab_plots, tab_audio, tab_harmonics, tab_help = st.tabs(
    ["📈 Plots", "🔊 Audio", "📊 Harmonics", "📘 How it works"]
)

with tab_plots:
    fig, axes = plotting.make_figure()
    plotting.render(axes, res, freqs, mag, show_recon)
    st.pyplot(fig, use_container_width=True)

    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)  # Prevent memory leaks across reruns

    st.download_button(
        "⬇️ Download plot (PNG)",
        data=png.getvalue(),
        file_name=f"plot_{kind}_{f}Hz_at_{fs}Hz.png",
        mime="image/png",
    )

with tab_audio:
    st.caption("Listen to how aliasing changes the perceived pitch and timbre. Lower your volume first.")
    col_orig, col_samp = st.columns(2, gap="large")

    with col_orig:
        with st.container(border=True):
            st.subheader("Original signal")
            st.caption(f"{kind.capitalize()} at {f:,} Hz")
            st.audio(cached_original_wav(kind, f), format="audio/wav")

    with col_samp:
        with st.container(border=True):
            st.subheader("Sampled & reconstructed")
            st.caption(f"fs = {fs:,} Hz · {'anti-aliased' if antialias else 'no filter'} · {recon.upper()}")
            samp_wav = cached_sampled_wav(kind, f, fs, antialias, recon)
            st.audio(samp_wav, format="audio/wav")
            st.download_button(
                label="⬇️ Download sampled WAV",
                data=samp_wav,
                file_name=f"sampled_{kind}_{f}Hz_at_{fs}Hz.wav",
                mime="audio/wav",
                use_container_width=True,
            )

with tab_harmonics:
    st.subheader(f"Harmonic fold-back (Nyquist = {nyquist:,.1f} Hz)")
    st.caption("A harmonic above Nyquist folds back and is heard at a different frequency. "
               "Orange rows are the ones that fold.")

    if kind == "sine":
        st.info("A sine wave has only one frequency, so there are no harmonics to fold. "
                "Pick the square or triangle wave to see this table in action.")

    def _status(r):
        if not r["aliased"]:
            return "✅ OK"
        return "🛡️ Removed by filter" if antialias else "⚠️ Folds back"

    df = pd.DataFrame(
        {
            "Harmonic": [r["k"] for r in harm_rows[:12]],
            "Frequency (Hz)": [r["freq"] for r in harm_rows[:12]],
            "Amplitude": [r["amp"] for r in harm_rows[:12]],
            "Appears at (Hz)": [r["alias"] for r in harm_rows[:12]],
            "Status": [_status(r) for r in harm_rows[:12]],
        }
    )

    def _highlight(row):
        color = "background-color: rgba(240,162,2,.22)" if row["Status"].startswith("⚠️") else ""
        return [color] * len(row)

    st.dataframe(
        df.style.apply(_highlight, axis=1).format(
            {"Frequency (Hz)": "{:,.1f}", "Amplitude": "{:.3f}", "Appears at (Hz)": "{:,.1f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    n_fold = sum(r["aliased"] for r in harm_rows)
    if kind != "sine" and n_fold and not antialias:
        st.warning(f"{n_fold} harmonic(s) are above Nyquist. Turn on the anti-aliasing filter to remove them.")
    st.caption("Showing the first 12 harmonics.")

with tab_help:
    st.markdown(
        """
**Nyquist rule:** to capture a signal without ambiguity, sample faster than twice its highest frequency (`fs > 2f`).

**Aliasing:** if `fs ≤ 2f`, the signal looks like a lower frequency: `f_alias = |f − fs · round(f / fs)|`.
Example: 900 Hz sampled at 1 kHz appears as 100 Hz.

**Anti-aliasing filter:** removes content above `fs/2` before sampling, so nothing can fold back.

**Reconstruction:** *sinc* is the ideal interpolation. *ZOH* holds each sample until the next, like a real DAC.

**Try this:**
1. Pick **Good sampling**, then raise f past 500 Hz and watch the banner flip.
2. Turn on the anti-aliasing filter and see the aliased tone disappear.
3. Pick the **Square** preset and open the Harmonics tab.
        """
    )