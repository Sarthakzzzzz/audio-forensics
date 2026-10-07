"""
Spectrogram generation module for Audio Forensic Analyzer.
Handles STFT-based spectrogram generation using Librosa.
"""

import numpy as np
import librosa
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SpectrogramAnalyzer:
    """Handles spectrogram generation and analysis."""

    def __init__(self, signal: np.ndarray, sample_rate: int):
        """
        Initialize spectrogram analyzer.

        Args:
            signal: Audio signal array
            sample_rate: Sample rate of the audio
        """
        self.signal = signal
        self.sample_rate = sample_rate
        self.spectrogram = None
        self.frequencies = None
        self.times = None

    def generate_spectrogram(
        self,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = 'hann',
        center: bool = True,
        pad_mode: str = 'reflect',
        power: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate spectrogram using Short-Time Fourier Transform.

        Args:
            n_fft: Length of the FFT window
            hop_length: Number of samples between successive frames
            win_length: Each frame of audio is windowed by window()
            window: Window function to use
            center: Whether to pad the signal to center frame
            pad_mode: Padding mode for centering
            power: Exponent for the magnitude spectrogram

        Returns:
            Tuple of (spectrogram, frequencies, times)
        """
        if hop_length is None:
            hop_length = n_fft // 4  # 75% overlap

        if win_length is None:
            win_length = n_fft

        # Compute STFT
        stft = librosa.stft(
            self.signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode
        )

        # Convert to power spectrogram
        spectrogram = np.abs(stft) ** power

        # Get frequencies and times
        self.frequencies = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)
        self.times = librosa.times_like(stft, sr=self.sample_rate, hop_length=hop_length, n_fft=n_fft)

        self.spectrogram = spectrogram
        return spectrogram, self.frequencies, self.times

    def get_spectrogram_db(self, ref: float = 1.0, amin: float = 1e-10, top_db: float = 80.0) -> np.ndarray:
        """
        Convert spectrogram to decibel scale.

        Args:
            ref: Reference value for power
            amin: Minimum amplitude to avoid log(0)
            top_db: Minimum negative cutoff in decibels

        Returns:
            Spectrogram in dB scale
        """
        if self.spectrogram is None:
            raise ValueError("Spectrogram not generated. Call generate_spectrogram first.")

        return librosa.power_to_db(self.spectrogram, ref=ref, amin=amin, top_db=top_db)

    def get_mel_spectrogram(
        self,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = 'hann',
        center: bool = True,
        pad_mode: str = 'reflect',
        power: float = 2.0,
        fmin: float = 0.0,
        fmax: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mel-spectrogram.

        Args:
            n_mels: Number of mel bands
            n_fft: Length of the FFT window
            hop_length: Number of samples between successive frames
            win_length: Each frame of audio is windowed by window()
            window: Window function to use
            center: Whether to pad the signal to center frame
            pad_mode: Padding mode for centering
            power: Exponent for the magnitude spectrogram
            fmin: Minimum frequency (Hz)
            fmax: Maximum frequency (Hz)

        Returns:
            Tuple of (mel_spectrogram, mel_frequencies, times)
        """
        if hop_length is None:
            hop_length = n_fft // 4

        if win_length is None:
            win_length = n_fft

        mel_spectrogram = librosa.feature.melspectrogram(
            y=self.signal,
            sr=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            power=power,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )

        mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
        times = librosa.times_like(mel_spectrogram, sr=self.sample_rate, hop_length=hop_length, n_fft=n_fft)

        return mel_spectrogram, mel_frequencies, times

    def get_spectral_centroid(self, n_fft: int = 2048, hop_length: Optional[int] = None) -> np.ndarray:
        """
        Calculate spectral centroid.

        Args:
            n_fft: Length of the FFT window
            hop_length: Number of samples between successive frames

        Returns:
            Spectral centroid values
        """
        if hop_length is None:
            hop_length = n_fft // 4

        return librosa.feature.spectral_centroid(
            y=self.signal,
            sr=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )[0]

    def get_spectral_rolloff(self, n_fft: int = 2048, hop_length: Optional[int] = None, roll_percent: float = 0.85) -> np.ndarray:
        """
        Calculate spectral rolloff.

        Args:
            n_fft: Length of the FFT window
            hop_length: Number of samples between successive frames
            roll_percent: Percentage of energy below which to compute rolloff

        Returns:
            Spectral rolloff values
        """
        if hop_length is None:
            hop_length = n_fft // 4

        return librosa.feature.spectral_rolloff(
            y=self.signal,
            sr=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            roll_percent=roll_percent
        )[0]


def create_spectrogram_analyzer(signal: np.ndarray, sample_rate: int) -> SpectrogramAnalyzer:
    """
    Factory function to create SpectrogramAnalyzer object.

    Args:
        signal: Audio signal array
        sample_rate: Sample rate of the audio

    Returns:
        SpectrogramAnalyzer object
    """
    return SpectrogramAnalyzer(signal, sample_rate)