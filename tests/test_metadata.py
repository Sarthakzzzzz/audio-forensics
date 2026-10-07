"""
Unit tests for metadata.py module
"""

import tempfile
import pytest
import numpy as np
import os
from forensic.metadata import analyze_metadata, MetadataAnalyzer


def test_metadata_analyzer_creation():
    """Test that MetadataAnalyzer object is created correctly."""
    # Create a temporary WAV file with minimal valid header
    wav_header = bytearray([
        0x52, 0x49, 0x46, 0x46,  # "RIFF"
        0x24, 0x00, 0x00, 0x00,  # size (36 bytes)
        0x57, 0x41, 0x56, 0x45,  # "WAVE"
        0x66, 0x6d, 0x74, 0x20,  # "fmt "
        0x10, 0x00, 0x00, 0x00,  # fmt chunk size (16)
        0x01, 0x00,              # audio format (1 = PCM)
        0x01, 0x00,              # channels (1 = mono)
        0x40, 0x1f, 0x00, 0x00,  # sample rate (8000)
        0x40, 0x1f, 0x00, 0x00,  # byte rate (8000)
        0x02, 0x00,              # block align (2)
        0x10, 0x00,              # bits per sample (16)
        0x64, 0x61, 0x74, 0x61,  # "data"
        0x00, 0x00, 0x00, 0x00   # data size (0)
    ])

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(wav_header)
        tmp_file_path = tmp_file.name

    try:
        # Create metadata analyzer
        analyzer = analyze_metadata(tmp_file_path)

        # Check that it's a MetadataAnalyzer instance
        assert isinstance(analyzer, MetadataAnalyzer)

        # Check that metadata is a dictionary
        assert isinstance(analyzer.metadata, dict)

        # Check for basic expected fields (may vary based on file)
        # At minimum, we should have duration, sample_rate, channels from info
        # Since our test file has no actual audio data, some might be None

    finally:
        os.unlink(tmp_file_path)


def test_metadata_anomalies_detection():
    """Test metadata anomaly detection."""
    # Create a minimal WAV file
    wav_header = bytearray([
        0x52, 0x49, 0x46, 0x46,  # "RIFF"
        0x24, 0x00, 0x00, 0x00,  # size (36 bytes)
        0x57, 0x41, 0x56, 0x45,  # "WAVE"
        0x66, 0x6d, 0x74, 0x20,  # "fmt "
        0x10, 0x00, 0x00, 0x00,  # fmt chunk size (16)
        0x01, 0x00,              # audio format (1 = PCM)
        0x01, 0x00,              # channels (1 = mono)
        0x40, 0x1f, 0x00, 0x00,  # sample rate (8000)
        0x40, 0x1f, 0x00, 0x00,  # byte rate (8000)
        0x02, 0x00,              # block align (2)
        0x10, 0x00,              # bits per sample (16)
        0x64, 0x61, 0x74, 0x61,  # "data"
        0x00, 0x00, 0x00, 0x00   # data size (0)
    ])

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(wav_header)
        tmp_file_path = tmp_file.name

    try:
        analyzer = MetadataAnalyzer(tmp_file_path)
        anomalies = analyzer.check_metadata_anomalies()

        # Check that anomalies is a list
        assert isinstance(anomalies, list)

        # Each anomaly should be a dictionary with expected keys
        for anomaly in anomalies:
            assert isinstance(anomaly, dict)
            assert 'type' in anomaly
            assert 'description' in anomaly

    finally:
        os.unlink(tmp_file_path)


def test_metadata_summary():
    """Test metadata summary generation."""
    wav_header = bytearray([
        0x52, 0x49, 0x46, 0x46,  # "RIFF"
        0x24, 0x00, 0x00, 0x00,  # size (36 bytes)
        0x57, 0x41, 0x56, 0x45,  # "WAVE"
        0x66, 0x6d, 0x74, 0x20,  # "fmt "
        0x10, 0x00, 0x00, 0x00,  # fmt chunk size (16)
        0x01, 0x00,              # audio format (1 = PCM)
        0x01, 0x00,              # channels (1 = mono)
        0x40, 0x1f, 0x00, 0x00,  # sample rate (8000)
        0x40, 0x1f, 0x00, 0x00,  # byte rate (8000)
        0x02, 0x00,              # block align (2)
        0x10, 0x00,              # bits per sample (16)
        0x64, 0x61, 0x74, 0x61,  # "data"
        0x00, 0x00, 0x00, 0x00   # data size (0)
    ])

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(wav_header)
        tmp_file_path = tmp_file.name

    try:
        analyzer = MetadataAnalyzer(tmp_file_path)
        summary = analyzer.get_metadata_summary()

        # Check summary structure
        assert isinstance(summary, dict)
        assert 'metadata' in summary
        assert 'anomalies' in summary
        assert 'is_reliable' in summary

        assert isinstance(summary['metadata'], dict)
        assert isinstance(summary['anomalies'], list)
        assert isinstance(summary['is_reliable'], bool)

    finally:
        os.unlink(tmp_file_path)


if __name__ == "__main__":
    pytest.test([__file__])