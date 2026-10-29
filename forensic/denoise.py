"""
Audio Denoising module for Audio Forensic Analyzer.
Provides noise reduction, filtering, and audio export capabilities.
"""

import io
import logging
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import scipy.signal
import soundfile as sf

logger = logging.getLogger(__name__)

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    logger.warning("noisereduce package not found, spectral subtraction fallback will be used.")


class AudioDenoiser:
    """Handles audio noise reduction, filtering, and export."""

    def __init__(self, signal: np.ndarray, sample_rate: int):
        """
        Initialize AudioDenoiser.

        Args:
            signal: Audio signal array (1D float)
            sample_rate: Sample rate in Hz
        """
        if signal.ndim > 1:
            # Average multi-channel down to mono
            signal = np.mean(signal, axis=1)

        self.original_signal = np.asarray(signal, dtype=np.float32)
        self.sample_rate = sample_rate
        self.denoised_signal = self.original_signal.copy()

    def reduce_noise(
        self,
        prop_decrease: float = 0.85,
        stationary: bool = True,
        time_constant_s: float = 2.0,
        freq_mask_smooth_hz: int = 500,
        time_mask_smooth_ms: int = 50,
        use_fallback: bool = False
    ) -> np.ndarray:
        """
        Perform noise reduction on audio signal using spectral gating.

        Args:
            prop_decrease: Noise reduction strength between 0.0 (no reduction) and 1.0 (maximum)
            stationary: Whether noise is stationary across the whole clip
            time_constant_s: Time constant for non-stationary noise
            freq_mask_smooth_hz: Frequency mask smoothing width in Hz
            time_mask_smooth_ms: Time mask smoothing width in ms
            use_fallback: Force use of internal spectral subtraction

        Returns:
            Denoised audio signal as 1D numpy array
        """
        prop_decrease = float(np.clip(prop_decrease, 0.0, 1.0))

        if prop_decrease == 0.0 or len(self.original_signal) == 0:
            self.denoised_signal = self.original_signal.copy()
            return self.denoised_signal

        if NOISEREDUCE_AVAILABLE and not use_fallback:
            try:
                cleaned = nr.reduce_noise(
                    y=self.original_signal,
                    sr=self.sample_rate,
                    prop_decrease=prop_decrease,
                    stationary=stationary,
                    time_constant_s=time_constant_s,
                    freq_mask_smooth_hz=freq_mask_smooth_hz,
                    time_mask_smooth_ms=time_mask_smooth_ms
                )
                self.denoised_signal = np.asarray(cleaned, dtype=np.float32)
                return self.denoised_signal
            except Exception as e:
                logger.warning(f"noisereduce failed: {e}. Falling back to internal spectral subtraction.")

        # Fallback spectral subtraction
        self.denoised_signal = self._spectral_subtraction(self.original_signal, prop_decrease=prop_decrease)
        return self.denoised_signal

    def _spectral_subtraction(
        self,
        signal: np.ndarray,
        prop_decrease: float = 0.85,
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Pure numpy/scipy spectral subtraction noise reduction fallback.
        Estimates noise from the lowest energy frames.
        """
        if len(signal) < n_fft:
            return signal.copy()

        # Compute STFT using scipy
        f, t, Zxx = scipy.signal.stft(signal, fs=self.sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Estimate noise spectrum from the bottom 15% lowest energy frames
        frame_energies = np.sum(magnitude ** 2, axis=0)
        num_noise_frames = max(1, int(0.15 * len(frame_energies)))
        noise_frame_indices = np.argsort(frame_energies)[:num_noise_frames]
        noise_profile = np.mean(magnitude[:, noise_frame_indices], axis=1, keepdims=True)

        # Spectral subtraction: |S| = max(|Y| - alpha * |N|, beta * |Y|)
        alpha = prop_decrease * 1.5
        beta = 0.05  # Spectral floor to prevent musical noise
        subtracted_magnitude = np.maximum(magnitude - alpha * noise_profile, beta * magnitude)

        # Reconstruct complex STFT and inverse STFT
        Zxx_cleaned = subtracted_magnitude * np.exp(1j * phase)
        _, cleaned_signal = scipy.signal.istft(Zxx_cleaned, fs=self.sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)

        # Match length of original signal
        if len(cleaned_signal) > len(signal):
            cleaned_signal = cleaned_signal[:len(signal)]
        elif len(cleaned_signal) < len(signal):
            cleaned_signal = np.pad(cleaned_signal, (0, len(signal) - len(cleaned_signal)))

        return np.asarray(cleaned_signal, dtype=np.float32)

    def apply_filter(
        self,
        filter_type: str = "none",
        low_cutoff: float = 80.0,
        high_cutoff: float = 4000.0,
        notch_freq: float = 50.0,
        order: int = 4,
        apply_to_denoised: bool = True
    ) -> np.ndarray:
        """
        Apply frequency domain filtering (bandpass, highpass, lowpass, notch).

        Args:
            filter_type: One of 'none', 'bandpass', 'highpass', 'lowpass', 'notch'
            low_cutoff: Lower cutoff frequency (Hz)
            high_cutoff: Upper cutoff frequency (Hz)
            notch_freq: Notch center frequency (Hz, typically 50 or 60 Hz for mains hum)
            order: Filter order
            apply_to_denoised: If True, filters the denoised signal; else original

        Returns:
            Filtered audio signal array
        """
        target = self.denoised_signal if apply_to_denoised else self.original_signal
        nyquist = 0.5 * self.sample_rate

        if filter_type.lower() == "none" or len(target) == 0:
            return target

        filtered = target.copy()

        try:
            if filter_type.lower() == "bandpass":
                low = max(10.0, min(low_cutoff, nyquist - 20)) / nyquist
                high = max(low_cutoff + 10, min(high_cutoff, nyquist - 10)) / nyquist
                if low < high:
                    sos = scipy.signal.butter(order, [low, high], btype='bandpass', output='sos')
                    filtered = scipy.signal.sosfiltfilt(sos, filtered)

            elif filter_type.lower() == "highpass":
                cutoff = max(10.0, min(low_cutoff, nyquist - 10)) / nyquist
                sos = scipy.signal.butter(order, cutoff, btype='highpass', output='sos')
                filtered = scipy.signal.sosfiltfilt(sos, filtered)

            elif filter_type.lower() == "lowpass":
                cutoff = max(50.0, min(high_cutoff, nyquist - 10)) / nyquist
                sos = scipy.signal.butter(order, cutoff, btype='lowpass', output='sos')
                filtered = scipy.signal.sosfiltfilt(sos, filtered)

            elif filter_type.lower() == "notch":
                if 10.0 < notch_freq < (nyquist - 10):
                    q = 30.0  # Quality factor
                    b, a = scipy.signal.iirnotch(notch_freq, q, fs=self.sample_rate)
                    filtered = scipy.signal.filtfilt(b, a, filtered)

            if apply_to_denoised:
                self.denoised_signal = np.asarray(filtered, dtype=np.float32)

            return np.asarray(filtered, dtype=np.float32)

        except Exception as e:
            logger.error(f"Filtering error ({filter_type}): {e}")
            return target

    def calculate_denoising_metrics(self) -> Dict[str, float]:
        """
        Calculate quantitative metrics comparing original (previous) and denoised (current) audio.

        Returns:
            Dictionary containing RMS values, SNR improvement, noise attenuation in dB, etc.
        """
        orig = self.original_signal
        denoised = self.denoised_signal

        rms_orig = float(np.sqrt(np.mean(orig ** 2))) if len(orig) > 0 else 0.0
        rms_denoised = float(np.sqrt(np.mean(denoised ** 2))) if len(denoised) > 0 else 0.0

        # Estimated noise removed: difference between original and denoised
        noise_diff = orig - denoised
        rms_noise = float(np.sqrt(np.mean(noise_diff ** 2))) if len(noise_diff) > 0 else 0.0

        # Peak amplitudes
        peak_orig = float(np.max(np.abs(orig))) if len(orig) > 0 else 0.0
        peak_denoised = float(np.max(np.abs(denoised))) if len(denoised) > 0 else 0.0

        # Estimated SNR improvement (dB)
        # Ratio of cleaned signal energy to residual difference noise energy
        eps = 1e-10
        if rms_noise > eps and rms_denoised > eps:
            snr_estimated = 20.0 * np.log10((rms_denoised + eps) / (rms_noise + eps))
        else:
            snr_estimated = 0.0

        # Noise power reduction (dB)
        if rms_orig > eps and rms_denoised > eps:
            noise_reduction_db = 20.0 * np.log10((rms_orig + eps) / (rms_denoised + eps))
        else:
            noise_reduction_db = 0.0

        return {
            "rms_original": rms_orig,
            "rms_denoised": rms_denoised,
            "rms_noise_removed": rms_noise,
            "peak_original": peak_orig,
            "peak_denoised": peak_denoised,
            "noise_reduction_db": max(0.0, float(noise_reduction_db)),
            "estimated_snr_gain_db": float(snr_estimated)
        }

    @staticmethod
    def to_wav_bytes(signal: np.ndarray, sample_rate: int) -> bytes:
        """
        Convert float audio array to standard PCM 16-bit WAV bytes for in-browser playback or download.

        Args:
            signal: Audio signal numpy array
            sample_rate: Sample rate in Hz

        Returns:
            WAV file bytes
        """
        if len(signal) == 0:
            return b""

        # Normalize to avoid clipping
        sig = signal.copy()
        max_val = np.max(np.abs(sig))
        if max_val > 1.0:
            sig = sig / max_val

        bio = io.BytesIO()
        sf.write(bio, sig, sample_rate, format='WAV', subtype='PCM_16')
        return bio.getvalue()


def create_denoiser(signal: np.ndarray, sample_rate: int) -> AudioDenoiser:
    """
    Factory function to create an AudioDenoiser instance.

    Args:
        signal: Audio signal array
        sample_rate: Sample rate in Hz

    Returns:
        AudioDenoiser instance
    """
    return AudioDenoiser(signal, sample_rate)

