"""
Demo script showing the core functionality of Audio Forensic Analyzer
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
from forensic.denoise import create_denoiser
from forensic.frequency import create_frequency_analyzer


def create_demo_audio_file(filename, duration=3.0, sample_rate=8000):
    """Create a demo WAV file with detectable changes."""
    # Generate test signal: sine wave that changes frequency midway
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # First half: 440 Hz, Second half: 880 Hz (to create a detectable change)
    # Add some noise to make it more realistic
    signal = np.where(t < duration/2,
                      np.sin(2 * np.pi * 440 * t),
                      np.sin(2 * np.pi * 880 * t))
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


def main():
    """Run the demo."""
    print("🎵 Audio Forensic Analyzer - Demo")
    print("=" * 50)

    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        test_file_path = tmp_file.name

    try:
        print(f"📁 Created test file: {os.path.basename(test_file_path)}")

        # Create test audio file with a detectable change
        create_demo_audio_file(test_file_path, duration=3.0, sample_rate=8000)
        print("🔊 Generated test audio with frequency change at 1.5 seconds")

        # Test 1: Evidence creation
        print("\n1️⃣  EVIDENCE ANALYSIS")
        evidence = create_evidence(test_file_path)
        evidence_data = evidence.get_evidence_summary()
        print(f"   Evidence ID: {evidence_data['evidence_id']}")
        print(f"   File Size: {evidence_data['file_size']:,} bytes")
        print(f"   SHA-256: {evidence_data['sha256'][:32]}...")

        # Test 2: Metadata analysis
        print("\n2️⃣  METADATA ANALYSIS")
        metadata_analyzer = analyze_metadata(test_file_path)
        metadata_results = metadata_analyzer.get_metadata_summary()
        metadata = metadata_results.get('metadata', {})
        print(f"   Format: {metadata.get('mime_type', 'Unknown')}")
        if metadata.get('duration'):
            print(f"   Duration: {metadata['duration']:.2f} seconds")
        if metadata.get('sample_rate'):
            print(f"   Sample Rate: {metadata['sample_rate']:,} Hz")
        if metadata.get('channels'):
            print(f"   Channels: {metadata['channels']}")

        # Test 3: Waveform loading
        print("\n3️⃣  WAVEFORM ANALYSIS")
        waveform_analyzer = load_audio_file(test_file_path)
        signal = waveform_analyzer.signal
        sample_rate = waveform_analyzer.sample_rate
        duration = len(signal) / sample_rate
        print(f"   Loaded {len(signal):,} samples")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Sample Rate: {sample_rate:,} Hz")

        # Test 4: Spectrogram generation
        print("\n4️⃣  SPECTROGRAM ANALYSIS")
        spectrogram_analyzer = create_spectrogram_analyzer(signal, sample_rate)
        spectrogram, frequencies, times = spectrogram_analyzer.generate_spectrogram()
        print(f"   Spectrogram shape: {spectrogram.shape}")
        print(f"   Frequency range: {frequencies[0]:.0f} - {frequencies[-1]:.0f} Hz")
        print(f"   Time range: {times[0]:.2f} - {times[-1]:.2f} seconds")

        # Test 5: Feature extraction
        print("\n5️⃣  FEATURE EXTRACTION")
        feature_extractor = create_feature_extractor(signal, sample_rate)
        features = feature_extractor.extract_all_features()
        print(f"   RMS energy: {np.mean(features['rms']):.4f}")
        print(f"   Zero crossing rate: {np.mean(features['zcr']):.4f}")
        print(f"   Spectral centroid: {np.mean(features['spectral_centroid']):.1f} Hz")
        print(f"   MFCC coefficients: {features['mfcc'].shape[0]} coefficients")

        # Test 6: Tampering detection
        print("\n6️⃣  TAMPERING DETECTION")
        tampering_detector = create_tampering_detector(
            window_size=0.5,
            overlap=0.5,
            anomaly_threshold=2.0
        )
        tampering_results = tampering_detector.detect_tampering(features, hop_length=512)
        print(f"   Analysis windows: {tampering_results['num_windows']}")
        print(f"   Suspicious regions: {len(tampering_results.get('suspicious_timestamps', []))}")
        if tampering_results.get('suspicious_timestamps'):
            print(f"   First suspicious timestamp: {tampering_results['suspicious_timestamps'][0]:.2f} seconds")
        print(f"   Max anomaly score: {tampering_results.get('max_anomaly_score', 0):.3f}")

        # Test 7: Audio Denoising
        print("\n7️⃣  AUDIO NOISE REDUCTION")
        denoiser = create_denoiser(signal, sample_rate)
        denoised_signal = denoiser.reduce_noise(prop_decrease=0.85, stationary=True)
        denoise_metrics = denoiser.calculate_denoising_metrics()
        print(f"   Noise reduction: {denoise_metrics['noise_reduction_db']:.2f} dB")
        print(f"   Estimated SNR gain: {denoise_metrics['estimated_snr_gain_db']:.2f} dB")
        print(f"   Original RMS: {denoise_metrics['rms_original']:.4f} -> Denoised RMS: {denoise_metrics['rms_denoised']:.4f}")

        # Test 8: Previous vs Current Frequency Change Analysis
        print("\n8️⃣  PREVIOUS VS CURRENT FREQUENCY CHANGE ANALYSIS")
        freq_analyzer = create_frequency_analyzer(sample_rate)
        freq_comp = freq_analyzer.compare_previous_and_current(signal, denoised_signal, jump_threshold_hz=150.0)
        f_metrics = freq_comp['metrics']
        print(f"   Previous Mean Centroid: {f_metrics['previous_mean_centroid_hz']} Hz")
        print(f"   Current Mean Centroid:  {f_metrics['current_mean_centroid_hz']} Hz (Change: {f_metrics['centroid_change_pct']:+.1f}%)")
        print(f"   High Freq Attenuation:  {f_metrics['high_freq_attenuation_db']:.1f} dB")
        print(f"   Detected Frequency Jumps: {len(freq_comp['detected_frequency_jumps'])}")
        for jump in freq_comp['detected_frequency_jumps'][:3]:
            print(f"     • At {jump['timestamp']:.2f}s: {jump['previous_frequency_hz']} Hz -> {jump['current_frequency_hz']} Hz (Δ={jump['frequency_change_hz']} Hz, {jump['direction']})")

        # Test 9: Report generation
        print("\n9️⃣  REPORT GENERATION")
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
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': 1,
                'bit_depth': 16,
                'bitrate': None,
                'codec': 'PCM'
            },
            metadata_results=metadata_results,
            waveform_data=(waveform_times, waveform_signal),
            spectrogram_data=(spectrogram, frequencies, times),
            tampering_results=report_tampering_results,
            denoise_metrics=denoise_metrics,
            frequency_comparison=freq_comp
        )

        # Check that report was created
        report_size = os.path.getsize(report_path)
        print(f"   Report generated: {os.path.basename(report_path)}")
        print(f"   Report size: {report_size:,} bytes")

        # Clean up report
        os.unlink(report_path)

        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        print("🔍 All core forensic analysis functions are working correctly.")
        print("📋 To run the full interactive application, execute:")
        print("   streamlit run app.py")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise

    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.unlink(test_file_path)


if __name__ == "__main__":
    main()