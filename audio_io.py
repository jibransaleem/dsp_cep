"""
audio_io.py  --  AUDIO DEVICE / FILE LAYER
===========================================
The only module that touches the sound card or the disk.
"""
import numpy as np
from scipy.io import wavfile

try:
    import sounddevice as sd
except Exception:          # no PortAudio / not installed
    sd = None


def audio_available():
    return sd is not None


def play(samples, fs):
    """Non-blocking playback. Raises RuntimeError if no audio backend."""
    if sd is None:
        raise RuntimeError("sounddevice is not installed.\n"
                           "Run:  pip install sounddevice\n"
                           "or use 'Save WAV' instead.")
    sd.stop()
    sd.play(np.asarray(samples, dtype=np.float32), fs)


def stop():
    if sd is not None:
        sd.stop()


def save_wav(path, samples, fs):
    wavfile.write(path, int(fs), (np.clip(samples, -1, 1) * 32767).astype(np.int16))