"""
Waveform visualization module for Audio Forensic Analyzer.
Handles audio loading and waveform generation for visualization.
"""

import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Optional
import logging

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

logger = logging.getLogger(__name__)


class WaveformAnalyzer:
    """Handles audio loading and waveform generation."""

    def __init__(self, file_path: str):
        """
        Initialize waveform analyzer.

        Args:
            file_path: Path to the audio file
        """
        self.file_path = file_path
        self.signal = None
        self.sample_rate = None
        self.duration = None
        self._load_audio()

    def _load_audio(self) -> None:
        """Load audio file using soundfile with librosa and PyAV fallbacks."""
        try:
            # Try soundfile first
            self.signal, self.sample_rate = sf.read(self.file_path)

            # Convert to mono if stereo for simplicity in some calculations
            if len(self.signal.shape) > 1:
                self.signal = np.mean(self.signal, axis=1)

        except Exception as e:
            logger.warning(f"Soundfile failed to load {self.file_path}: {e}")
            try:
                # Fallback to librosa
                self.signal, self.sample_rate = librosa.load(
                    self.file_path, sr=None, mono=True
                )
            except Exception as e2:
                logger.warning(f"Librosa also failed to load {self.file_path}: {e2}")
                # Fallback to PyAV for WebM, Opus, and container formats
                if AV_AVAILABLE:
                    try:
                        self.signal, self.sample_rate = self._load_with_av(self.file_path)
                    except Exception as e3:
                        logger.error(f"PyAV failed to load {self.file_path}: {e3}")
                        raise ValueError(f"Could not load audio file: {self.file_path}")
                else:
                    raise ValueError(f"Could not load audio file: {self.file_path}")

        # Calculate duration
        if self.signal is not None and self.sample_rate is not None:
            self.duration = len(self.signal) / self.sample_rate

    def _load_with_av(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file using PyAV (supports WebM, Opus, etc.)."""
        container = av.open(file_path)
        audio_streams = container.streams.audio
        if not audio_streams:
            raise ValueError(f"No audio stream found in {file_path}")

        stream = audio_streams[0]
        sample_rate = stream.sample_rate

        frames = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray())

        if not frames:
            raise ValueError(f"Audio stream contained no frames in {file_path}")

        # Concatenate frames along time axis (axis 1: shape is (channels, samples))
        audio = np.concatenate(frames, axis=1)

        # Convert to float32 normalized between -1.0 and 1.0 if integer dtype
        if np.issubdtype(audio.dtype, np.integer):
            max_val = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float32) / max_val
        else:
            audio = audio.astype(np.float32)

        # Downmix multi-channel to mono
        if audio.ndim > 1:
            if audio.shape[0] > 1:
                audio = np.mean(audio, axis=0)
            else:
                audio = audio[0]

        return audio, sample_rate

    def get_waveform_data(self, downsample_factor: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get waveform data for plotting.

        Args:
            downsample_factor: Factor to downsample for performance (None for auto)

        Returns:
            Tuple of (time_array, amplitude_array)
        """
        if self.signal is None or self.sample_rate is None:
            raise ValueError("Audio not loaded")

        # Create time array
        times = np.arange(len(self.signal)) / self.sample_rate

        # Downsample for performance if needed
        if downsample_factor is None:
            # Auto downsample if signal is too long (>50k points)
            if len(self.signal) > 50000:
                downsample_factor = max(1, len(self.signal) // 50000)
            else:
                downsample_factor = 1

        if downsample_factor > 1:
            # Downsample both signal and times
            indices = np.arange(0, len(self.signal), downsample_factor)
            signal_downsampled = self.signal[indices]
            times_downsampled = times[indices]
            return times_downsampled, signal_downsampled
        else:
            return times, self.signal

    def get_normalized_waveform(self) -> np.ndarray:
        """Get normalized waveform amplitude (-1 to 1)."""
        if self.signal is None:
            raise ValueError("Audio not loaded")

        # Normalize to [-1, 1] range
        if np.max(np.abs(self.signal)) > 0:
            return self.signal / np.max(np.abs(self.signal))
        else:
            return self.signal

    def get_audio_properties(self) -> dict:
        """Get basic audio properties."""
        return {
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'num_samples': len(self.signal) if self.signal is not None else 0,
            'channels': 1  # We convert to mono for analysis
        }


def load_audio_file(file_path: str) -> WaveformAnalyzer:
    """
    Factory function to create WaveformAnalyzer object.

    Args:
        file_path: Path to the audio file

    Returns:
        WaveformAnalyzer object
    """
    return WaveformAnalyzer(file_path)