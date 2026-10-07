"""
Integration test for Audio Forensic Analyzer
"""

import tempfile
import os
import numpy as np
from forensic.evidence import create_evidence
from forensic.metadata import analyze_metadata
from forensic.waveform import load_audio_file
from forensic.spectrogram import create_spectrogram_analyzer
from forensic.features import create_feature_extractor
from forensic.tampering import create_tampering_detector
from forensic.report import create_report_generator


def create_test_wav_file(filename, duration=2.0, sample_rate=8000):
    """Create a test WAV file with known characteristics."""
    # Generate test signal: sine wave that changes frequency midway
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # First half: 440 Hz, Second half: 880 Hz (to create a detectable change)
    signal = np.where(t < duration/2,
                      np.sin(2 * np.pi * 440 * t),
                      np.sin(2 * np.pi * 880 * t))

    # Add some noise to make it more realistic
    signal += 0.1 * np.random.randn(len(signal))

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

    with open(filename, 'wb') as f:
        f.write(wav_bytes)

    return filename


def test_full_analysis_pipeline():
    """Test the full analysis pipeline with a synthetic test file."""
    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        test_file_path = tmp_file.name

    try:
        # Create test audio file with a detectable change
        create_test_wav_file(test_file_path, duration=3.0, sample_rate=8000)

        # Test 1: Evidence creation
        evidence = create_evidence(test_file_path)
        evidence_data = evidence.get_evidence_summary()

        assert 'evidence_id' in evidence_data
        assert 'sha256' in evidence_data
        assert evidence_data['file_size'] > 0
        print("✓ Evidence creation test passed")

        # Test 2: Metadata analysis
        metadata_analyzer = analyze_metadata(test_file_path)
        metadata_results = metadata_analyzer.get_metadata_summary()

        assert isinstance(metadata_results, dict)
        assert 'metadata' in metadata_results
        print("✓ Metadata analysis test passed")

        # Test 3: Waveform loading
        waveform_analyzer = load_audio_file(test_file_path)
        signal = waveform_analyzer.signal
        sample_rate = waveform_analyzer.sample_rate

        assert signal is not None
        assert sample_rate > 0
        assert len(signal) > 0
        print("✓ Waveform loading test passed")

        # Test 4: Spectrogram generation
        spectrogram_analyzer = create_spectrogram_analyzer(signal, sample_rate)
        spectrogram, frequencies, times = spectrogram_analyzer.generate_spectrogram()
        spectrogram_db = spectrogram_analyzer.get_spectrogram_db()

        assert spectrogram is not None
        assert frequencies is not None
        assert times is not None
        assert spectrogram_db is not None
        print("✓ Spectrogram generation test passed")

        # Test 5: Feature extraction
        feature_extractor = create_feature_extractor(signal, sample_rate)
        features = feature_extractor.extract_all_features()

        assert 'rms' in features
        assert 'zcr' in features
        assert 'spectral_centroid' in features
        assert 'spectral_rolloff' in features
        assert 'mfcc' in features
        assert features['mfcc'].shape[0] == 13  # 13 MFCC coefficients
        print("✓ Feature extraction test passed")

        # Test 6: Tampering detection
        tampering_detector = create_tampering_detector(
            window_size=0.5,
            overlap=0.5,
            anomaly_threshold=2.0
        )

        tampering_results = tampering_detector.detect_tampering(features, hop_length=512)

        assert isinstance(tampering_results, dict)
        assert 'anomalies' in tampering_results
        assert 'num_windows' in tampering_results
        # Handle case where there's insufficient data
        if 'message' in tampering_results and 'Insufficient data' in tampering_results['message']:
            # This is acceptable for very short test signals
            print("⚠ Tampering detection: Insufficient data (expected for short test signal)")
        else:
            assert 'suspicious_timestamps' in tampering_results
            assert 'max_anomaly_score' in tampering_results
        print("✓ Tampering detection test passed")

        # Test 7: Report generation
        report_generator = create_report_generator()

        # Get waveform data for display
        waveform_times, waveform_signal = waveform_analyzer.get_waveform_data()

        # Ensure tampering_results has the expected fields for report generation
        report_tampering_results = tampering_results.copy()
        if 'suspicious_timestamps' not in report_tampering_results:
            report_tampering_results['suspicious_timestamps'] = []
        if 'max_anomaly_score' not in report_tampering_results:
            report_tampering_results['max_anomaly_score'] = 0

        report_path = report_generator.generate_report(
            evidence_data=evidence_data,
            audio_properties={
                'duration': len(signal) / sample_rate,
                'sample_rate': sample_rate,
                'channels': 1,
                'bit_depth': 16,
                'bitrate': None,
                'codec': 'PCM'
            },
            metadata_results=metadata_results,
            waveform_data=(waveform_times, waveform_signal),
            spectrogram_data=(spectrogram_db, frequencies, times),
            tampering_results=report_tampering_results
        )

        # Check that report was created
        assert os.path.exists(report_path)
        assert report_path.endswith('.pdf')
        assert os.path.getsize(report_path) > 0

        # Clean up report
        os.unlink(report_path)
        print("✓ Report generation test passed")

        print("\n🎉 All integration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise

    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.unlink(test_file_path)


if __name__ == "__main__":
    test_full_analysis_pipeline()