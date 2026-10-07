"""
Feature extraction module for Audio Forensic Analyzer.
Extracts basic audio features for anomaly detection.
"""

import numpy as np
import librosa
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts audio features for forensic analysis."""

    def __init__(self, signal: np.ndarray, sample_rate: int):
        """
        Initialize feature extractor.

        Args:
            signal: Audio signal array
            sample_rate: Sample rate of the audio
        """
        self.signal = signal
        self.sample_rate = sample_rate

    def extract_rms_energy(self, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Extract Root Mean Square energy.

        Args:
            frame_length: Length of each frame
            hop_length: Number of samples between frames

        Returns:
            RMS energy values
        """
        return librosa.feature.rms(
            y=self.signal,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

    def extract_zero_crossing_rate(self, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Extract Zero Crossing Rate.

        Args:
            frame_length: Length of each frame
            hop_length: Number of samples between frames

        Returns:
            Zero crossing rate values
        """
        return librosa.feature.zero_crossing_rate(
            y=self.signal,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

    def extract_spectral_centroid(self, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Extract Spectral Centroid.

        Args:
            frame_length: Length of each frame
            hop_length: Number of samples between frames

        Returns:
            Spectral centroid values
        """
        return librosa.feature.spectral_centroid(
            y=self.signal,
            sr=self.sample_rate,
            n_fft=frame_length,
            hop_length=hop_length
        )[0]

    def extract_spectral_rolloff(self, frame_length: int = 2048, hop_length: int = 512, roll_percent: float = 0.85) -> np.ndarray:
        """
        Extract Spectral Rolloff.

        Args:
            frame_length: Length of each frame
            hop_length: Number of samples between frames
            roll_percent: Percentage of energy below which to compute rolloff

        Returns:
            Spectral rolloff values
        """
        return librosa.feature.spectral_rolloff(
            y=self.signal,
            sr=self.sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
            roll_percent=roll_percent
        )[0]

    def extract_mfcc(self, n_mfcc: int = 13, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Extract Mel-frequency cepstral coefficients.

        Args:
            n_mfcc: Number of MFCC coefficients to extract
            frame_length: Length of each frame
            hop_length: Number of samples between frames

        Returns:
            MFCC coefficients (n_mfcc x t)
        """
        return librosa.feature.mfcc(
            y=self.signal,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=frame_length,
            hop_length=hop_length
        )

    def extract_all_features(self, frame_length: int = 2048, hop_length: int = 512) -> dict:
        """
        Extract all basic audio features.

        Args:
            frame_length: Length of each frame
            hop_length: Number of samples between frames

        Returns:
            Dictionary containing all extracted features
        """
        features = {
            'rms': self.extract_rms_energy(frame_length=frame_length, hop_length=hop_length),
            'zcr': self.extract_zero_crossing_rate(frame_length=frame_length, hop_length=hop_length),
            'spectral_centroid': self.extract_spectral_centroid(frame_length=frame_length, hop_length=hop_length),
            'spectral_rolloff': self.extract_spectral_rolloff(frame_length=frame_length, hop_length=hop_length),
            'mfcc': self.extract_mfcc(n_mfcc=13, frame_length=frame_length, hop_length=hop_length)
        }

        return features


def create_feature_extractor(signal: np.ndarray, sample_rate: int) -> FeatureExtractor:
    """
    Factory function to create FeatureExtractor object.

    Args:
        signal: Audio signal array
        sample_rate: Sample rate of the audio

    Returns:
        FeatureExtractor object
    """
    return FeatureExtractor(signal, sample_rate)