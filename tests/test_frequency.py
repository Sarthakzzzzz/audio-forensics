"""
Unit tests for FrequencyAnalyzer module.
"""

import numpy as np
import pytest
from forensic.frequency import create_frequency_analyzer, FrequencyAnalyzer


@pytest.fixture
def frequency_shift_signal():
    """Create a signal that shifts frequency midway from 440 Hz to 880 Hz with noise."""
    sr = 8000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), False)
    signal = np.where(t < 1.0, np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t))
    signal = signal + 0.1 * np.random.randn(len(t))
    return signal.astype(np.float32), sr


def test_frequency_analyzer_creation():
    """Test creating FrequencyAnalyzer instance."""
    analyzer = create_frequency_analyzer(8000)
    assert isinstance(analyzer, FrequencyAnalyzer)
    assert analyzer.sample_rate == 8000


def test_power_spectrum(frequency_shift_signal):
    """Test computing power spectrum (PSD)."""
    signal, sr = frequency_shift_signal
    analyzer = create_frequency_analyzer(sr)
    freqs, psd_db = analyzer.compute_power_spectrum(signal)

    assert len(freqs) > 0
    assert len(psd_db) == len(freqs)
    assert freqs[0] == 0.0
    assert freqs[-1] <= sr / 2


def test_temporal_frequency_tracking(frequency_shift_signal):
    """Test temporal frequency tracking and jump detection."""
    signal, sr = frequency_shift_signal
    analyzer = create_frequency_analyzer(sr)
    tracking = analyzer.compute_temporal_frequency_tracking(signal, jump_threshold_hz=150.0)

    assert "times" in tracking
    assert "spectral_centroid" in tracking
    assert "dominant_frequency" in tracking
    assert "delta_frequency" in tracking
    assert "frequency_jumps" in tracking

    # Check that frequency jump near 1.0s is detected
    jumps = tracking["frequency_jumps"]
    assert len(jumps) > 0
    # At least one jump should be near 1.0s (between 0.8s and 1.2s)
    near_one_sec = [j for j in jumps if 0.8 <= j["timestamp"] <= 1.2]
    assert len(near_one_sec) > 0


def test_compare_previous_and_current(frequency_shift_signal):
    """Test comprehensive comparison of previous vs current audio."""
    signal, sr = frequency_shift_signal
    # Simulate a cleaned signal with slightly attenuated high frequencies
    denoised_signal = signal * 0.9

    analyzer = create_frequency_analyzer(sr)
    comp = analyzer.compare_previous_and_current(signal, denoised_signal)

    assert "spectrum_comparison" in comp
    assert "temporal_comparison" in comp
    assert "metrics" in comp
    assert "detected_frequency_jumps" in comp

    metrics = comp["metrics"]
    assert "previous_mean_centroid_hz" in metrics
    assert "current_mean_centroid_hz" in metrics
    assert "centroid_change_pct" in metrics
    assert "previous_mean_rolloff_hz" in metrics
    assert "current_mean_rolloff_hz" in metrics

