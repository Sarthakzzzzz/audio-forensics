"""
Metadata analysis module for Audio Forensic Analyzer.
Extracts and analyzes audio file metadata using Mutagen.
"""

from typing import Dict, Any, Optional, List
import mutagen
from mutagen.id3 import ID3
from mutagen.mp4 import MP4
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE
import logging

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetadataAnalyzer:
    """Analyzes audio file metadata."""

    def __init__(self, file_path: str):
        """
        Initialize metadata analyzer.

        Args:
            file_path: Path to the audio file
        """
        self.file_path = file_path
        self.metadata = self._extract_metadata()

    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from audio file."""
        metadata = {}

        try:
            audio_file = mutagen.File(self.file_path)
            if audio_file is None:
                if AV_AVAILABLE:
                    av_meta = self._extract_av_metadata()
                    if av_meta:
                        return av_meta
                logger.warning(f"Could not read metadata from {self.file_path}")
                return metadata

            # Handle different file formats
            if isinstance(audio_file, ID3):  # MP3
                metadata.update(self._extract_id3_metadata(audio_file))
            elif isinstance(audio_file, MP4):  # M4A/MP4
                metadata.update(self._extract_mp4_metadata(audio_file))
            elif isinstance(audio_file, FLAC):  # FLAC
                metadata.update(self._extract_flac_metadata(audio_file))
            elif isinstance(audio_file, OggVorbis):  # OGG
                metadata.update(self._extract_ogg_metadata(audio_file))
            elif isinstance(audio_file, WAVE):  # WAV
                metadata.update(self._extract_wav_metadata(audio_file))
            else:
                # Generic metadata extraction
                metadata.update(self._extract_generic_metadata(audio_file))

            # Add technical properties that are usually available
            if audio_file.info:
                metadata.update({
                    'duration': getattr(audio_file.info, 'length', None),
                    'sample_rate': getattr(audio_file.info, 'sample_rate', None),
                    'channels': getattr(audio_file.info, 'channels', None),
                    'bitrate': getattr(audio_file.info, 'bitrate', None),
                })

                # Try to get bit depth if available
                if hasattr(audio_file.info, 'bits_per_sample'):
                    metadata['bit_depth'] = audio_file.info.bits_per_sample

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            metadata['error'] = str(e)

        return metadata

    def _extract_id3_metadata(self, audio_file: ID3) -> Dict[str, Any]:
        """Extract ID3 metadata (MP3)."""
        metadata = {}
        tag_mapping = {
            'TIT2': 'title',
            'TPE1': 'artist',
            'TALB': 'album',
            'TCON': 'genre',
            'TYER': 'year',
            'COMM': 'comment',
            'TRCK': 'track_number'
        }

        for tag_id, tag_name in tag_mapping.items():
            if tag_id in audio_file:
                metadata[tag_name] = str(audio_file[tag_id])

        return metadata

    def _extract_mp4_metadata(self, audio_file: MP4) -> Dict[str, Any]:
        """Extract MP4 metadata (M4A/MP4)."""
        metadata = {}
        tag_mapping = {
            '\xa9nam': 'title',
            '\xa9ART': 'artist',
            '\xa9alb': 'album',
            '\xa9gen': 'genre',
            '\xa9day': 'year',
            '\xa9cmt': 'comment',
            'trkn': 'track_number'
        }

        for tag_id, tag_name in tag_mapping.items():
            if tag_id in audio_file:
                value = audio_file[tag_id]
                if isinstance(value, list) and len(value) > 0:
                    metadata[tag_name] = str(value[0])
                else:
                    metadata[tag_name] = str(value)

        return metadata

    def _extract_flac_metadata(self, audio_file: FLAC) -> Dict[str, Any]:
        """Extract FLAC metadata."""
        metadata = {}
        tag_mapping = {
            'title': 'title',
            'artist': 'artist',
            'album': 'album',
            'genre': 'genre',
            'date': 'year',
            'comment': 'comment',
            'tracknumber': 'track_number'
        }

        for tag_id, tag_name in tag_mapping.items():
            if tag_id in audio_file:
                metadata[tag_name] = str(audio_file[tag_id])

        return metadata

    def _extract_ogg_metadata(self, audio_file: OggVorbis) -> Dict[str, Any]:
        """Extract OGG Vorbis metadata."""
        metadata = {}
        tag_mapping = {
            'title': 'title',
            'artist': 'artist',
            'album': 'album',
            'genre': 'genre',
            'date': 'year',
            'comment': 'comment',
            'tracknumber': 'track_number'
        }

        for tag_id, tag_name in tag_mapping.items():
            if tag_id in audio_file:
                metadata[tag_name] = str(audio_file[tag_id])

        return metadata

    def _extract_wav_metadata(self, audio_file: WAVE) -> Dict[str, Any]:
        """Extract WAV metadata (limited)."""
        metadata = {}
        # WAV files typically don't have extensive metadata
        # but we can check for INFO chunk
        if hasattr(audio_file, 'tags') and audio_file.tags:
            for key, value in audio_file.tags.items():
                metadata[key.lower()] = str(value)

        return metadata

    def _extract_av_metadata(self) -> Dict[str, Any]:
        """Extract metadata using PyAV for formats like WebM and Opus."""
        metadata = {}
        try:
            container = av.open(self.file_path)
            metadata['format'] = container.format.name
            audio_streams = container.streams.audio
            if audio_streams:
                stream = audio_streams[0]
                metadata['sample_rate'] = stream.sample_rate
                metadata['channels'] = stream.channels
                if hasattr(stream.codec_context, 'name') and stream.codec_context.name:
                    metadata['codec'] = stream.codec_context.name

                # Duration
                if container.duration:
                    metadata['duration'] = float(container.duration) / av.time_base
                elif stream.duration and stream.time_base:
                    metadata['duration'] = float(stream.duration * stream.time_base)

                if container.bit_rate and container.bit_rate > 0:
                    metadata['bitrate'] = container.bit_rate
                elif stream.bit_rate and stream.bit_rate > 0:
                    metadata['bitrate'] = stream.bit_rate

            # Container metadata tags
            for key, value in container.metadata.items():
                metadata[key.lower()] = str(value)
        except Exception as e:
            logger.error(f"PyAV metadata extraction error: {e}")
        return metadata

    def _extract_generic_metadata(self, audio_file) -> Dict[str, Any]:
        """Extract generic metadata."""
        metadata = {}
        if hasattr(audio_file, 'tags') and audio_file.tags:
            for key, value in audio_file.tags.items():
                metadata[str(key)] = str(value)

        return metadata

    def check_metadata_anomalies(self) -> List[Dict[str, Any]]:
        """
        Check for metadata anomalies.

        Returns:
            List of anomaly dictionaries
        """
        anomalies = []

        # Check for missing essential metadata
        essential_fields = ['duration', 'sample_rate', 'channels']
        for field in essential_fields:
            if field not in self.metadata or self.metadata[field] is None:
                anomalies.append({
                    'type': 'missing_metadata',
                    'field': field,
                    'description': f'Essential metadata field "{field}" is missing'
                })

        # Check for suspicious values
        if 'duration' in self.metadata and self.metadata['duration']:
            duration = self.metadata['duration']
            if duration < 0:
                anomalies.append({
                    'type': 'invalid_value',
                    'field': 'duration',
                    'value': duration,
                    'description': 'Negative duration detected'
                })
            elif duration > 24 * 3600:  # More than 24 hours
                anomalies.append({
                    'type': 'suspicious_value',
                    'field': 'duration',
                    'value': duration,
                    'description': 'Unusually long duration (>24 hours)'
                })

        if 'sample_rate' in self.metadata and self.metadata['sample_rate']:
            sample_rate = self.metadata['sample_rate']
            if sample_rate <= 0:
                anomalies.append({
                    'type': 'invalid_value',
                    'field': 'sample_rate',
                    'value': sample_rate,
                    'description': 'Non-positive sample rate detected'
                })
            elif sample_rate > 384000:  # Unusually high sample rate
                anomalies.append({
                    'type': 'suspicious_value',
                    'field': 'sample_rate',
                    'value': sample_rate,
                    'description': 'Unusually high sample rate detected'
                })

        return anomalies

    def get_metadata_summary(self) -> Dict[str, Any]:
        """Get metadata summary."""
        return {
            'metadata': self.metadata,
            'anomalies': self.check_metadata_anomalies(),
            'is_reliable': len(self.check_metadata_anomalies()) == 0
        }


def analyze_metadata(file_path: str) -> MetadataAnalyzer:
    """
    Factory function to create MetadataAnalyzer object.

    Args:
        file_path: Path to the audio file

    Returns:
        MetadataAnalyzer object
    """
    return MetadataAnalyzer(file_path)