"""
Unit tests for WebM audio support.
"""

import os
import pytest
from forensic.waveform import load_audio_file, WaveformAnalyzer
from forensic.metadata import analyze_metadata
from forensic.evidence import create_evidence


def test_webm_file_loading():
    """Test loading and decoding a WebM audio file."""
    sample_path = "./archive/sample-1.webm"
    if not os.path.exists(sample_path):
        pytest.skip("sample-1.webm not found")

    # 1. Evidence
    evidence = create_evidence(sample_path)
    assert evidence.evidence_id.startswith("EVID_")
    assert evidence.sha256_hash is not None

    # 2. Metadata
    meta = analyze_metadata(sample_path)
    summary = meta.get_metadata_summary()
    assert summary["metadata"].get("sample_rate") == 48000
    assert summary["metadata"].get("codec") == "opus"

    # 3. Waveform loading & decoding
    analyzer = load_audio_file(sample_path)
    assert isinstance(analyzer, WaveformAnalyzer)
    assert analyzer.signal is not None
    assert len(analyzer.signal) > 0
    assert analyzer.sample_rate == 48000
    assert analyzer.duration > 0

