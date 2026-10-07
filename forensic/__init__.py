"""
Audio Forensic Analyzer - Forensic Package
"""

# Version information
__version__ = "1.0.0"
__author__ = "Audio Forensic Analyzer Team"
__description__ = "Lightweight digital forensic analysis tool for audio evidence"

# Import main classes and functions for easy access
from .evidence import create_evidence, Evidence
from .metadata import analyze_metadata, MetadataAnalyzer
from .waveform import load_audio_file, WaveformAnalyzer
from .spectrogram import create_spectrogram_analyzer, SpectrogramAnalyzer
from .features import create_feature_extractor, FeatureExtractor
from .tampering import create_tampering_detector, TamperingDetector
from .report import create_report_generator, PDFReportGenerator

# Define what gets imported with "from forensic import *"
__all__ = [
    "create_evidence",
    "Evidence",
    "analyze_metadata",
    "MetadataAnalyzer",
    "load_audio_file",
    "WaveformAnalyzer",
    "create_spectrogram_analyzer",
    "SpectrogramAnalyzer",
    "create_feature_extractor",
    "FeatureExtractor",
    "create_tampering_detector",
    "TamperingDetector",
    "create_report_generator",
    "PDFReportGenerator"
]