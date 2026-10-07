"""
Unit tests for evidence.py module
"""

import os
import tempfile
import hashlib
import pytest
from forensic.evidence import create_evidence, Evidence


def test_evidence_creation():
    """Test that Evidence object is created correctly."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(b"test audio data")
        tmp_file_path = tmp_file.name

    try:
        # Create evidence object
        evidence = create_evidence(tmp_file_path)

        # Check that it's an Evidence instance
        assert isinstance(evidence, Evidence)

        # Check evidence ID is generated
        assert evidence.evidence_id is not None
        assert len(evidence.evidence_id) > 0
        assert evidence.evidence_id.startswith("EVID_")

        # Check file info
        assert evidence.file_info['filename'] == os.path.basename(tmp_file_path)
        assert evidence.file_info['file_size'] == len(b"test audio data")

        # Check SHA-256 hash
        expected_hash = hashlib.sha256(b"test audio data").hexdigest()
        assert evidence.sha256_hash == expected_hash

    finally:
        # Clean up
        os.unlink(tmp_file_path)


def test_evidence_summary():
    """Test evidence summary generation."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(b"test data for summary")
        tmp_file_path = tmp_file.name

    try:
        evidence = create_evidence(tmp_file_path)
        summary = evidence.get_evidence_summary()

        # Check summary contains expected fields
        assert 'evidence_id' in summary
        assert 'filename' in summary
        assert 'file_size' in summary
        assert 'mime_type' in summary
        assert 'sha256' in summary
        assert 'analysis_timestamp' in summary

        # Check values
        assert summary['filename'] == os.path.basename(tmp_file_path)
        assert summary['file_size'] == len(b"test data for summary")
        assert summary['sha256'] == hashlib.sha256(b"test data for summary").hexdigest()

    finally:
        os.unlink(tmp_file_path)


def test_nonexistent_file():
    """Test that nonexistent file raises appropriate error."""
    with pytest.raises(FileNotFoundError):
        create_evidence("/nonexistent/file.wav")


def test_sha256_consistency():
    """Test that SHA-256 is consistent for same file."""
    test_data = b"This is a test for SHA-256 consistency"

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(test_data)
        tmp_file_path = tmp_file.name

    try:
        evidence1 = create_evidence(tmp_file_path)
        evidence2 = create_evidence(tmp_file_path)

        # Hashes should be identical
        assert evidence1.sha256_hash == evidence2.sha256_hash

    finally:
        os.unlink(tmp_file_path)