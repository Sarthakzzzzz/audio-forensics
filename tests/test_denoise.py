"""
Unit tests for AudioDenoiser module.
"""

import numpy as np
import pytest
from forensic.denoise import create_denoiser, AudioDenoiser


@pytest.fixture
def sample_noisy_signal():
    """Create a sample noisy sine wave."""
    sr = 8000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), False)
    signal = np.sin(2 * np.pi * 440 * t) + 0.2 * np.random.randn(len(t))
    return signal.astype(np.float32), sr


def test_denoiser_creation(sample_noisy_signal):
    """Test creating AudioDenoiser instance."""
    signal, sr = sample_noisy_signal
    denoiser = create_denoiser(signal, sr)
    assert isinstance(denoiser, AudioDenoiser)
    assert denoiser.sample_rate == sr
    assert len(denoiser.original_signal) == len(signal)


def test_noise_reduction(sample_noisy_signal):
    """Test reducing noise from audio signal."""
    signal, sr = sample_noisy_signal
    denoiser = create_denoiser(signal, sr)
    cleaned = denoiser.reduce_noise(prop_decrease=0.8, stationary=True)

    assert len(cleaned) == len(signal)
    assert isinstance(cleaned, np.ndarray)
    # Energy of cleaned signal should be less than noisy signal due to removed noise
    orig_energy = np.mean(signal ** 2)
    cleaned_energy = np.mean(cleaned ** 2)
    assert cleaned_energy < orig_energy


def test_fallback_spectral_subtraction(sample_noisy_signal):
    """Test the pure-scipy fallback spectral subtraction method."""
    signal, sr = sample_noisy_signal
    denoiser = create_denoiser(signal, sr)
    cleaned = denoiser.reduce_noise(prop_decrease=0.8, use_fallback=True)

    assert len(cleaned) == len(signal)
    assert np.isfinite(cleaned).all()


def test_filtering_methods(sample_noisy_signal):
    """Test audio filters (bandpass, highpass, lowpass, notch)."""
    signal, sr = sample_noisy_signal
    denoiser = create_denoiser(signal, sr)

    bp = denoiser.apply_filter("bandpass", low_cutoff=200, high_cutoff=1000)
    assert len(bp) == len(signal)
    assert np.isfinite(bp).all()

    hp = denoiser.apply_filter("highpass", low_cutoff=100)
    assert len(hp) == len(signal)
    assert np.isfinite(hp).all()

    lp = denoiser.apply_filter("lowpass", high_cutoff=1000)
    assert len(lp) == len(signal)
    assert np.isfinite(lp).all()

    notch = denoiser.apply_filter("notch", notch_freq=50)
    assert len(notch) == len(signal)
    assert np.isfinite(notch).all()


def test_denoising_metrics(sample_noisy_signal):
    """Test metrics calculation between original and denoised signal."""
    signal, sr = sample_noisy_signal
    denoiser = create_denoiser(signal, sr)
    denoiser.reduce_noise(prop_decrease=0.8)
    metrics = denoiser.calculate_denoising_metrics()

    assert "rms_original" in metrics
    assert "rms_denoised" in metrics
    assert "rms_noise_removed" in metrics
    assert "noise_reduction_db" in metrics
    assert metrics["rms_original"] > 0
    assert metrics["rms_noise_removed"] > 0


def test_to_wav_bytes(sample_noisy_signal):
    """Test exporting signal to in-memory WAV bytes."""
    signal, sr = sample_noisy_signal
    wav_bytes = AudioDenoiser.to_wav_bytes(signal, sr)

    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 44  # Standard WAV header size
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"

