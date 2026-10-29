"""
Frequency Analysis module for Audio Forensic Analyzer.
Tracks frequency changes over time and compares previous (original) vs current (denoised) frequency properties.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import librosa
import scipy.signal

logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    """Analyzes and compares audio frequency content and tracks temporal frequency shifts."""

    def __init__(self, sample_rate: int):
        """
        Initialize FrequencyAnalyzer.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate

    def compute_power_spectrum(
        self,
        signal: np.ndarray,
        n_fft: int = 2048
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute average power spectral density (PSD in dB) across all frequencies.

        Args:
            signal: Audio signal array
            n_fft: FFT window size

        Returns:
            Tuple of (frequencies in Hz, magnitude in dB)
        """
        if len(signal) == 0:
            return np.array([]), np.array([])

        if len(signal) < n_fft:
            n_fft = 2 ** int(np.floor(np.log2(max(16, len(signal)))))

        freqs, psd = scipy.signal.welch(
            signal,
            fs=self.sample_rate,
            nperseg=n_fft,
            scaling='density'
        )

        # Convert to dB relative to peak
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))
        return freqs, psd_db

    def compute_temporal_frequency_tracking(
        self,
        signal: np.ndarray,
        frame_length: int = 2048,
        hop_length: int = 512,
        jump_threshold_hz: float = 150.0
    ) -> Dict[str, Any]:
        """
        Track instantaneous frequency characteristics and frame-to-frame frequency changes across time.

        Args:
            signal: Audio signal array
            frame_length: STFT frame length
            hop_length: STFT hop length
            jump_threshold_hz: Threshold in Hz to classify a frequency change as a significant jump

        Returns:
            Dictionary containing:
                - times: Array of timestamps in seconds
                - spectral_centroid: Centroid frequency (Hz) at each frame
                - dominant_frequency: Peak frequency (Hz) at each frame
                - delta_frequency: Frame-to-frame change |f_t - f_{t-1}| in Hz
                - frequency_jumps: List of detected abrupt frequency transitions with timestamps
        """
        if len(signal) < frame_length:
            return {
                "times": np.array([0.0]),
                "spectral_centroid": np.array([0.0]),
                "dominant_frequency": np.array([0.0]),
                "delta_frequency": np.array([0.0]),
                "frequency_jumps": []
            }

        # 1. Spectral Centroid over time
        centroid = librosa.feature.spectral_centroid(
            y=signal,
            sr=self.sample_rate,
            n_fft=frame_length,
            hop_length=hop_length
        )[0]

        # Time stamps for frames
        times = librosa.frames_to_time(
            np.arange(len(centroid)),
            sr=self.sample_rate,
            hop_length=hop_length
        )

        # 2. Dominant (peak) frequency per frame from STFT
        stft = np.abs(librosa.stft(signal, n_fft=frame_length, hop_length=hop_length))
        fft_freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=frame_length)
        dominant_indices = np.argmax(stft, axis=0)
        dominant_freqs = fft_freqs[dominant_indices]

        # 3. Step-to-step frequency change (delta f) between previous and current frame
        delta_centroid = np.abs(np.diff(centroid, prepend=centroid[0]))
        delta_dominant = np.abs(np.diff(dominant_freqs, prepend=dominant_freqs[0]))

        # Detect abrupt frequency jump events (splice/tone transitions)
        frequency_jumps = []
        for i in range(1, len(times)):
            d_c = float(delta_centroid[i])
            d_d = float(delta_dominant[i])
            if d_d >= jump_threshold_hz or d_c >= jump_threshold_hz:
                prev_f = float(dominant_freqs[i - 1])
                curr_f = float(dominant_freqs[i])
                shift = curr_f - prev_f
                frequency_jumps.append({
                    "timestamp": round(float(times[i]), 3),
                    "frame_index": i,
                    "previous_frequency_hz": round(prev_f, 1),
                    "current_frequency_hz": round(curr_f, 1),
                    "frequency_change_hz": round(abs(shift), 1),
                    "direction": "up" if shift > 0 else "down"
                })

        return {
            "times": times,
            "spectral_centroid": centroid,
            "dominant_frequency": dominant_freqs,
            "delta_frequency": delta_dominant,
            "delta_centroid": delta_centroid,
            "frequency_jumps": frequency_jumps
        }

    def compare_previous_and_current(
        self,
        original_signal: np.ndarray,
        denoised_signal: np.ndarray,
        frame_length: int = 2048,
        hop_length: int = 512,
        jump_threshold_hz: float = 150.0
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison of frequency characteristics between Previous (Original) and Current (Denoised) audio.

        Args:
            original_signal: Original (previous) audio signal
            denoised_signal: Denoised (current) audio signal
            frame_length: Frame length for analysis
            hop_length: Hop length for analysis
            jump_threshold_hz: Sensitivity threshold for frequency jump detection

        Returns:
            Dictionary containing:
                - spectrum_comparison: {freqs, orig_db, denoised_db, attenuation_db}
                - temporal_comparison: {times, orig_centroid, denoised_centroid, orig_dominant, denoised_dominant}
                - metrics: Scalar statistics for previous vs current frequency profiles
                - detected_jumps: Frequency jump events detected in previous vs current
        """
        # 1. Global Frequency Spectrum (Welch PSD)
        freqs, orig_psd_db = self.compute_power_spectrum(original_signal, n_fft=frame_length)
        _, denoised_psd_db = self.compute_power_spectrum(denoised_signal, n_fft=frame_length)

        attenuation_db = orig_psd_db - denoised_psd_db if len(orig_psd_db) == len(denoised_psd_db) else np.array([])

        # 2. Temporal tracking for both
        orig_tracking = self.compute_temporal_frequency_tracking(
            original_signal, frame_length, hop_length, jump_threshold_hz=jump_threshold_hz
        )
        denoised_tracking = self.compute_temporal_frequency_tracking(
            denoised_signal, frame_length, hop_length, jump_threshold_hz=jump_threshold_hz
        )

        # 3. Overall Spectral Features
        orig_centroid = orig_tracking["spectral_centroid"]
        denoised_centroid = denoised_tracking["spectral_centroid"]

        # Spectral Rolloff
        rolloff_orig = librosa.feature.spectral_rolloff(
            y=original_signal, sr=self.sample_rate, n_fft=frame_length, hop_length=hop_length
        )[0]
        rolloff_denoised = librosa.feature.spectral_rolloff(
            y=denoised_signal, sr=self.sample_rate, n_fft=frame_length, hop_length=hop_length
        )[0]

        # Spectral Bandwidth
        bw_orig = librosa.feature.spectral_bandwidth(
            y=original_signal, sr=self.sample_rate, n_fft=frame_length, hop_length=hop_length
        )[0]
        bw_denoised = librosa.feature.spectral_bandwidth(
            y=denoised_signal, sr=self.sample_rate, n_fft=frame_length, hop_length=hop_length
        )[0]

        # Spectral Flatness (tonality indicator - lower means more harmonic, less noisy)
        flatness_orig = librosa.feature.spectral_flatness(
            y=original_signal, n_fft=frame_length, hop_length=hop_length
        )[0]
        flatness_denoised = librosa.feature.spectral_flatness(
            y=denoised_signal, n_fft=frame_length, hop_length=hop_length
        )[0]

        mean_orig_centroid = float(np.mean(orig_centroid)) if len(orig_centroid) > 0 else 0.0
        mean_denoised_centroid = float(np.mean(denoised_centroid)) if len(denoised_centroid) > 0 else 0.0
        centroid_change_pct = (
            ((mean_denoised_centroid - mean_orig_centroid) / (mean_orig_centroid + 1e-6)) * 100.0
        )

        mean_orig_rolloff = float(np.mean(rolloff_orig)) if len(rolloff_orig) > 0 else 0.0
        mean_denoised_rolloff = float(np.mean(rolloff_denoised)) if len(rolloff_denoised) > 0 else 0.0
        rolloff_change_pct = (
            ((mean_denoised_rolloff - mean_orig_rolloff) / (mean_orig_rolloff + 1e-6)) * 100.0
        )

        mean_orig_flatness = float(np.mean(flatness_orig)) if len(flatness_orig) > 0 else 0.0
        mean_denoised_flatness = float(np.mean(flatness_denoised)) if len(flatness_denoised) > 0 else 0.0

        # High-frequency noise suppression (> 3000 Hz or > 0.5 Nyquist)
        high_freq_idx = np.where(freqs > min(3000.0, 0.5 * (self.sample_rate / 2)))[0]
        if len(high_freq_idx) > 0 and len(attenuation_db) > 0:
            avg_high_freq_attenuation_db = float(np.mean(attenuation_db[high_freq_idx]))
        else:
            avg_high_freq_attenuation_db = 0.0

        return {
            "spectrum_comparison": {
                "frequencies": freqs,
                "original_psd_db": orig_psd_db,
                "denoised_psd_db": denoised_psd_db,
                "attenuation_db": attenuation_db
            },
            "temporal_comparison": {
                "times": orig_tracking["times"],
                "original_centroid": orig_centroid,
                "denoised_centroid": denoised_centroid,
                "original_dominant": orig_tracking["dominant_frequency"],
                "denoised_dominant": denoised_tracking["dominant_frequency"],
                "delta_original": orig_tracking["delta_frequency"],
                "delta_denoised": denoised_tracking["delta_frequency"]
            },
            "metrics": {
                "previous_mean_centroid_hz": round(mean_orig_centroid, 1),
                "current_mean_centroid_hz": round(mean_denoised_centroid, 1),
                "centroid_change_pct": round(centroid_change_pct, 2),
                "previous_mean_rolloff_hz": round(mean_orig_rolloff, 1),
                "current_mean_rolloff_hz": round(mean_denoised_rolloff, 1),
                "rolloff_change_pct": round(rolloff_change_pct, 2),
                "previous_flatness": round(mean_orig_flatness, 5),
                "current_flatness": round(mean_denoised_flatness, 5),
                "high_freq_attenuation_db": round(max(0.0, avg_high_freq_attenuation_db), 2)
            },
            "detected_frequency_jumps": orig_tracking["frequency_jumps"]
        }


def create_frequency_analyzer(sample_rate: int) -> FrequencyAnalyzer:
    """
    Factory function to create FrequencyAnalyzer instance.

    Args:
        sample_rate: Sample rate in Hz

    Returns:
        FrequencyAnalyzer instance
    """
    return FrequencyAnalyzer(sample_rate)
