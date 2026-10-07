"""
Unit tests for features.py module
"""

import tempfile
import pytest
import numpy as np
import os
from forensic.features import create_feature_extractor, FeatureExtractor
from forensic.waveform import load_audio_file


def test_feature_extractor_creation():
    """Test that FeatureExtractor object is created correctly."""
    # Create a simple test signal
    signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 8000))  # 1 second of 440Hz
    sample_rate = 8000

    extractor = create_feature_extractor(signal, sample_rate)

    # Check that it's a FeatureExtractor instance
    assert isinstance(extractor, FeatureExtractor)
    assert np.array_equal(extractor.signal, signal)
    assert extractor.sample_rate == sample_rate


def test_rms_energy_extraction():
    """Test RMS energy extraction."""
    # Create test signal
    signal = np.ones(8000) * 0.5  # Constant signal
    sample_rate = 8000

    extractor = create_feature_extractor(signal, sample_rate)
    rms = extractor.extract_rms_energy()

    # For constant signal of 0.5, RMS should be approximately 0.5
    # (exact value depends on framing implementation)
    expected_rms = 0.5
    # Use a reasonable tolerance for RMS calculation
    np.testing.assert_allclose(np.mean(rms), expected_rms, rtol=0.05)


def test_zero_crossing_rate_extraction():
    """Test Zero Crossing Rate extraction."""
    # Create signal that crosses zero frequently
    # Using a proper alternating signal as float
    signal = np.tile([1.0, -1.0], 1000)  # Alternating 1.0, -1.0
    sample_rate = 8000

    extractor = create_feature_extractor(signal, sample_rate)
    zcr = extractor.extract_zero_crossing_rate()

    # For alternating 1.0, -1.0, we expect high zero crossing rate
    # But due to framing, it might not be exactly 1.0
    assert np.mean(zcr) > 0.5  # Should be reasonably high


def test_spectral_centroid_extraction():
    """Test spectral centroid extraction."""
    # Create a simple test signal
    signal = np.random.randn(8000)  # White noise
    sample_rate = 8000

    extractor = create_feature_extractor(signal, sample_rate)
    centroid = extractor.extract_spectral_centroid()

    # Should return array of reasonable length
    assert len(centroid) > 0
    # Values should be positive and reasonable for sample rate
    assert np.all(centroid >= 0)
    assert np.all(centroid <= sample_rate // 2)  # Nyquist frequency


def test_spectral_rolloff_extraction():
    """Test spectral rolloff extraction."""
    signal = np.random.randn(8000)
    sample_rate = 8000

    extractor = create_feature_extractor(signal, sample_rate)
    rolloff = extractor.extract_spectral_rolloff()

    # Should return array of reasonable length
    assert len(rolloff) > 0
    # Values should be positive and reasonable
    assert np.all(rolloff >= 0)
    assert np.all(rolloff <= sample_rate // 2)


def test_mfcc_extraction():
    """Test MFCC extraction."""
    signal = np.random.randn(8000)
    sample_rate = 8000

    extractor = create_feature_extractor(signal, sample_rate)
    mfcc = extractor.extract_mfcc(n_mfcc=13)

    # Should return 13 coefficients
    assert mfcc.shape[0] == 13
    assert mfcc.shape[1] > 0  # Should have time frames


def test_extract_all_features():
    """Test extraction of all features."""
    signal = np.random.randn(8000)
    sample_rate = 8000

    extractor = create_feature_extractor(signal, sample_rate)
    features = extractor.extract_all_features()

    # Check that all expected features are present
    expected_features = ['rms', 'zcr', 'spectral_centroid', 'spectral_rolloff', 'mfcc']
    for feature in expected_features:
        assert feature in features

    # Check shapes
    assert len(features['rms']) > 0
    assert len(features['zcr']) > 0
    assert len(features['spectral_centroid']) > 0
    assert len(features['spectral_rolloff']) > 0
    assert features['mfcc'].shape[0] == 13  # 13 MFCC coefficients


def test_feature_extractor_with_real_audio():
    """Test feature extraction with a real audio file."""
    # Create a minimal WAV file for testing
    # RIFF header for 1 second of 8kHz mono 16-bit PCM
    sample_rate = 8000
    duration = 1.0
    num_samples = int(sample_rate * duration)

    # Generate a simple sine wave
    t = np.linspace(0, duration, num_samples, False)
    signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

    # Convert to 16-bit PCM
    audio_data = (signal * 32767).astype(np.int16)

    # WAV header
    wav_header = bytearray()
    wav_header.extend(b'RIFF')  # Chunk ID
    wav_header.extend((36 + len(audio_data) * 2).to_bytes(4, 'little'))  # Chunk size
    wav_header.extend(b'WAVE')  # Format
    wav_header.extend(b'fmt ')  # Subchunk1 ID
    wav_header.extend((16).to_bytes(4, 'little'))  # Subchunk1 size (16 for PCM)
    wav_header.extend((1).to_bytes(2, 'little'))  # Audio format (1 for PCM)
    wav_header.extend((1).to_bytes(2, 'little'))  # Num channels (1 for mono)
    wav_header.extend(sample_rate.to_bytes(4, 'little'))  # Sample rate
    wav_header.extend((sample_rate * 2).to_bytes(4, 'little'))  # Byte rate
    wav_header.extend((2).to_bytes(2, 'little'))  # Block align
    wav_header.extend((16).to_bytes(2, 'little'))  # Bits per sample
    wav_header.extend(b'data')  # Subchunk2 ID
    wav_header.extend((len(audio_data) * 2).to_bytes(4, 'little'))  # Subchunk2 size

    # Combine header and data
    wav_bytes = bytes(wav_header) + audio_data.tobytes()

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(wav_bytes)
        tmp_file_path = tmp_file.name

    try:
        # Load audio using waveform analyzer
        waveform_analyzer = load_audio_file(tmp_file_path)
        signal_loaded = waveform_analyzer.signal
        sample_rate_loaded = waveform_analyzer.sample_rate

        # Extract features
        extractor = create_feature_extractor(signal_loaded, sample_rate_loaded)
        features = extractor.extract_all_features()

        # Basic sanity checks
        assert len(features['rms']) > 0
        assert len(features['zcr']) > 0
        assert len(features['spectral_centroid']) > 0
        assert len(features['spectral_rolloff']) > 0
        assert features['mfcc'].shape[0] == 13

    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


if __name__ == "__main__":
    pytest.main([__file__])