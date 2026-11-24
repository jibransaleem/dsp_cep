"""Unit tests for the pure DSP logic (run: python -m unittest -v)."""
import unittest
import numpy as np
import dsp_core as d


class TestAlias(unittest.TestCase):
    def test_folding(self):
        self.assertAlmostEqual(float(d.alias_frequency(900, 1000)), 100)
        self.assertAlmostEqual(float(d.alias_frequency(100, 1000)), 100)
        self.assertAlmostEqual(float(d.alias_frequency(1100, 1000)), 100)
        self.assertAlmostEqual(float(d.alias_frequency(500, 1000)), 500)


class TestSampling(unittest.TestCase):
    def test_sine_perfect_reconstruction_below_nyquist(self):
        r = d.run_sampling("sine", 100, 1000, n_cycles=6)
        mid = slice(1000, 3000)          # ignore edge effects of finite sinc sum
        err = np.max(np.abs(r.x_recon[mid] - r.x_dense[mid]))
        self.assertLess(err, 0.05)
        self.assertFalse(r.aliased)

    def test_aliased_samples_match_alias_tone(self):
        # samples of a 900 Hz sine at 1000 Hz equal samples of a -100 Hz sine
        r = d.run_sampling("sine", 900, 1000, n_cycles=6)
        expected = d.waveform("sine", -100, r.t_s)
        np.testing.assert_allclose(r.x_s, expected, atol=1e-9)
        self.assertTrue(r.aliased)
        self.assertAlmostEqual(r.f_alias, 100)

    def test_spectrum_peak_at_alias(self):
        fr, mag = d.spectrum_of_sampled("sine", 900, 1000)
        self.assertAlmostEqual(fr[np.argmax(mag)], 100, delta=1.0)

    def test_antialias_removes_alias(self):
        fr, mag = d.spectrum_of_sampled("sine", 900, 1000, antialias=True)
        self.assertLess(mag.max(), 1e-9)

    def test_bandlimited_square_matches_ideal_when_fs_high(self):
        t = np.linspace(0.001, 0.009, 50)           # avoid jump points
        bl = d.bandlimited_waveform("square", 100, t, 200000)
        np.testing.assert_allclose(bl, d.waveform("square", 100, t), atol=0.05)

    def test_triangle_series_matches_ideal(self):
        t = np.linspace(0, 0.02, 200)
        bl = d.bandlimited_waveform("triangle", 100, t, 50000)
        np.testing.assert_allclose(bl, d.waveform("triangle", 100, t), atol=1e-3)

    def test_zoh_is_staircase(self):
        r = d.run_sampling("sine", 50, 500, recon="zoh")
        self.assertEqual(set(np.round(r.x_recon, 12)), set(np.round(r.x_s, 12)))

    def test_harmonic_table_flags(self):
        rows = d.harmonic_table("square", 300, 2000)
        self.assertFalse(rows[0]["aliased"])         # 300 Hz OK
        self.assertTrue(rows[-1]["aliased"])         # high harmonics fold


class TestAudio(unittest.TestCase):
    def test_sampled_audio_pitch_is_alias(self):
        y = d.synth_sampled("sine", 900, 1000)       # should be heard at 100 Hz
        spec = np.abs(np.fft.rfft(y))
        peak_hz = np.argmax(spec) * d.AUDIO_FS / len(y)
        self.assertAlmostEqual(peak_hz, 100, delta=2)


if __name__ == "__main__":
    unittest.main()