# Audio Forensic Analyzer

A lightweight digital forensic analysis tool for audio evidence, inspired by the workflow and visualization concepts of Sonic Visualiser but implemented independently in Python.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow Example](#workflow-example)
- [Explanation of Key Concepts](#explanation-of-key-concepts)
- [Limitations](#limitations)
- [Forensic Disclaimer](#forensic-disclaimer)
- [Future Improvements](#future-improvements)
- [Testing](#testing)
- [Docker Support](#docker-support)

## Overview

The Audio Forensic Analyzer is a college-level mini-project designed for preliminary digital forensic examination of audio evidence. It provides tools for evidence acquisition, metadata analysis, waveform and spectrogram visualization, basic feature extraction, and tampering detection through anomaly analysis.

**Important Forensic Principles Implemented:**
1. Original evidence is never modified
2. Evidence is hashed before analysis for integrity verification
3. Analysis is reproducible and transparent
4. Every suspicious finding includes a timestamp
5. Anomaly detection indicates potential issues, not definitive proof
6. Metadata alone is not treated as proof of authenticity
7. Reports clearly describe methods used
8. Original evidence is kept separate from generated artifacts

## Features

### 1. Evidence Acquisition
- Secure file upload without modifying original evidence
- Unique Evidence ID generation
- Immediate SHA-256 hash calculation for integrity verification
- File information capture (size, MIME type, timestamps)
- Support for WAV, MP3, FLAC, OGG, M4A formats

### 2. Metadata Analysis
- Extraction of technical and textual metadata using Mutagen
- Display of format, duration, sample rate, channels, bitrate, codec
- ID3, MP4, FLAC, OGG Vorbis, and WAV metadata support
- Metadata anomaly detection (missing fields, suspicious values)
- Clear labeling of metadata as "reported" (not inherently trustworthy)

### 3. Waveform Visualization
- Interactive amplitude-over-time display
- X-axis = time in seconds, Y-axis = normalized amplitude
- Zoom capabilities for detailed inspection
- Performance-optimized downsampling for long files
- Duration display and readable scaling

### 4. Spectrogram Analysis
- STFT-based spectrogram generation using Librosa
- Time (X-axis) vs Frequency (Y-axis) with magnitude in dB
- Logarithmic frequency scaling option
- Adjustable FFT size and hop length
- Interactive inspection of suspicious regions

### 5. Basic Signal Features
- RMS energy calculation
- Zero Crossing Rate (ZCR)
- Spectral Centroid
- Spectral Rolloff
- 13 MFCC coefficients
- Features used internally for anomaly detection

### 6. Tampering / Splice Anomaly Detection
- Window-based analysis (default 1-second windows with 50% overlap)
- Feature discontinuity analysis between adjacent windows
- Robust threshold-based anomaly detection
- Optional Isolation Forest machine learning approach
- Output includes suspicious timestamps, anomaly scores, and feature changes
- Clear distinction between "suspicious" and "proven manipulation"

### 7. Forensic Findings Dashboard
- Evidence integrity verification display
- Metadata findings presentation
- Signal analysis confirmation
- Suspicious regions enumeration
- Overall assessment with appropriate uncertainty language

### 8. PDF Forensic Report Generation
- Comprehensive report with all analysis results
- Embedded waveform and spectrogram images
- Methodology documentation
- Forensic disclaimer inclusion
- SHA-256 hash and evidence tracking
- Professional formatting suitable for documentation

## Architecture

```
audio-forensics/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── Dockerfile                # Docker container configuration
├── .gitignore               # Git ignore rules
│
├── forensic/                 # Core forensic analysis modules
│   ├── __init__.py           # Package initializer
│   ├── evidence.py           # Evidence handling and hashing
│   ├── metadata.py           # Metadata extraction and analysis
│   ├── waveform.py           # Audio loading and waveform generation
│   ├── spectrogram.py        # STFT and spectrogram generation
│   ├── features.py           # Audio feature extraction (MFCC, RMS, etc.)
│   ├── tampering.py          # Anomaly detection and tampering analysis
│   └── report.py             # PDF report generation
│
├── reports/                  # Generated PDF reports directory
├── samples/                  # Test audio files directory
└── tests/                    # Unit tests
    ├── test_evidence.py
    ├── test_metadata.py
    ├── test_features.py
    └── test_tampering.py
```

### Module Responsibilities

- **evidence.py**: SHA-256 hashing, file information, MIME detection, Evidence ID generation
- **metadata.py**: Metadata extraction using Mutagen, metadata normalization, anomaly checks
- **waveform.py**: Audio loading, waveform generation, downsampling for visualization performance
- **spectrogram.py**: STFT implementation, spectrogram generation, dB conversion, spectral features
- **features.py**: MFCC, RMS, ZCR, spectral centroid, spectral rolloff extraction
- **tampering.py**: Window segmentation, feature comparison, anomaly scoring, suspicious timestamp detection
- **report.py**: PDF report creation, formatting, image embedding
- **app.py**: Streamlit UI orchestration, user interaction, analysis workflow coordination

## Technology Stack

- **Python 3.11+**: Core programming language
- **Streamlit**: Interactive web-based user interface
- **Librosa**: Audio analysis, feature extraction, spectrogram generation
- **NumPy**: Numerical computations and array operations
- **SciPy**: Scientific computing algorithms
- **SoundFile**: Audio file I/O operations
- **Mutagen**: Metadata extraction from audio formats
- **python-magic**: File type identification and MIME detection
- **Scikit-learn**: Optional Isolation Forest for anomaly detection
- **Plotly**: Interactive visualizations (available for extension)
- **ReportLab**: PDF report generation
- **hashlib**: Cryptographic SHA-256 hashing

## Installation

### Local Installation

1. **Clone the repository** (or extract the project files):
   ```bash
   git clone <repository-url>
   cd audio-forensics
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
   ```

### Docker Installation

1. **Build the Docker image**:
   ```bash
   docker build -t audio-forensic-analyzer .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8501:8501 audio-forensic-analyzer
   ```

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Basic Workflow

1. **Upload an Audio File**
   - Use the file uploader in the sidebar
   - Supported formats: WAV, MP3, FLAC, OGG, M4A
   - The original file is never modified

2. **Review Evidence Information**
   - Examine the Evidence ID, filename, file size, and MIME type
   - Verify the SHA-256 hash for integrity confirmation
   - Note the analysis timestamp

3. **Adjust Analysis Settings** (Optional)
   - Window Size: Analysis window duration in seconds (0.5-5.0)
   - Window Overlap: Percentage overlap between windows (0.0-0.9)
   - Anomaly Threshold: Sensitivity for detection (1.0-5.0 standard deviations)
   - Use Isolation Forest: Enable machine learning-based detection
   - Contamination: Expected outlier proportion for Isolation Forest

4. **Analyze Results**
   - **Audio Properties**: Duration, sample rate, channels, bit depth, bitrate, codec
   - **Metadata**: Extracted tags and technical information
   - **Waveform**: Interactive amplitude-over-time visualization
   - **Spectrogram**: Time-frequency representation with dB scaling
   - **Tampering Analysis**: Suspicious regions, anomaly scores, detection statistics

5. **Generate PDF Report**
   - Click "Generate PDF Report" button
   - Download the comprehensive forensic analysis report
   - Report includes all visualizations, findings, and methodology

## Workflow Example

Here's an example of how to use the tool for analyzing an audio file:

1. **Upload** a suspicious audio file named `evidence001.wav`
2. **Observe** the Evidence ID: `EVID_20261007_143022_123456`
3. **Verify** the SHA-256 hash: `a1b2c3d4...` (64-character hexadecimal)
4. **Review** metadata showing sample rate 44100 Hz, stereo channels
5. **Examine** waveform for clipping or unusual patterns
6. **Inspect** spectrogram for horizontal lines (indicating cuts) or frequency gaps
7. **Check** tampering analysis for suspicious regions:
   - Possible audio discontinuity detected at 18.42 seconds
   - Anomaly score: 0.81
8. **Generate** PDF report for documentation and further analysis

## Explanation of Key Concepts

### Waveform
The waveform displays audio amplitude over time. The X-axis represents time in seconds, and the Y-axis shows normalized amplitude values (-1 to 1). This visualization helps identify:
- Clipping (distortion at maximum amplitude)
- Silent regions
- Sudden amplitude changes (potential edits)
- Overall signal structure

### Spectrogram
The spectrogram shows how the frequency content of the signal changes over time:
- X-axis: Time (seconds)
- Y-axis: Frequency (Hz)
- Color intensity: Magnitude/power of each frequency component (in dB)
This visualization helps identify:
- Frequency-based edits or replacements
- Continuous tones vs. speech/music
- Background noise characteristics
- Sudden frequency domain changes

### MFCC (Mel-frequency Cepstral Coefficients)
MFCCs represent the short-term power spectrum of sound, based on a linear cosine transform of a log power spectrum on a nonlinear Mel scale of frequency. They are particularly useful for:
- Characterizing timbral properties
- Detecting changes in audio texture
- Identifying potential splicing points
- Comparing audio segments for similarity

### Anomaly Detection
The tampering detection algorithm works by:
1. Dividing audio into overlapping windows (default 1-second)
2. Extracting features (RMS, ZCR, spectral centroid, rolloff, MFCCs) for each window
3. Comparing features between adjacent windows
4. Calculating discontinuity scores based on feature differences
5. Flagging windows with unusually large changes as suspicious

**Important**: Anomaly detection indicates potential issues requiring further investigation, not definitive proof of manipulation.

## Limitations

As a college mini-project, this tool has intentional limitations:

1. **Scope**: Focused on preliminary analysis only, not comprehensive forensic examination
2. **No Real-time Processing**: Designed for file-based analysis, not live audio
3. **Limited Algorithms**: Uses basic threshold-based and optional ML detection
4. **No Advanced Features**: 
   - No speaker identification
   - No deepfake detection
   - No ENF (Electrical Network Frequency) analysis
   - No steganography detection
   - No MIDI processing
   - No plugin system
5. **Simplified Models**: Anomaly detection uses heuristic approaches
6. **Educational Focus**: Intended for learning and preliminary investigation

## Forensic Disclaimer

> **FORENSIC DISCLAIMER**: This tool provides preliminary digital forensic analysis. Detection of an anomaly does not by itself establish that an audio recording was manipulated. Findings should be corroborated using additional forensic methods and appropriate reference evidence. This analysis is intended for educational and preliminary investigative purposes only. The tool implements basic anomaly detection algorithms that may produce false positives or miss sophisticated manipulations. Always consult with qualified digital forensic examiners for legal or official investigations.

## Future Improvements

Potential enhancements for future development:

1. **Enhanced Visualization**: 
   - Interactive zoom and pan in Plotly visualizations
   - Region selection and analysis
   - Overlay comparison of multiple files

2. **Advanced Detection Methods**:
   - Machine learning models trained on known tampered/authentic audio
   - Statistical approaches for more robust anomaly detection
   - Frequency-domain specific tampering detection

3. **Additional Features**:
   - Batch processing capabilities
   - Comparison mode for analyzing multiple files
   - Audio enhancement/filters for better visualization
   - Timeline-based evidence correlation

4. **Report Enhancements**:
   - Customizable report templates
   - Chain of custody documentation
   - Export to multiple formats (HTML, JSON, etc.)

5. **Performance Improvements**:
   - Better caching mechanisms
   - Progressive loading for very large files
   - Multiprocessing for intensive computations

6. **Testing and Validation**:
   - Expanded test suite with more edge cases
   - Validation against known test datasets
   - Performance benchmarking

## Testing

### Running Unit Tests

The project includes basic unit tests for core functionality:

```bash
# From the project root directory
python -m pytest tests/ -v
```

### Test Coverage

Tests cover:
- SHA-256 calculation accuracy
- File identification and MIME detection
- Metadata extraction from various formats
- Audio feature extraction (MFCC, RMS, ZCR, etc.)
- Anomaly scoring algorithms
- Suspicious timestamp detection
- Synthetic test audio generation and verification

### Synthetic Test Generation

Tests include synthetic audio generation to create controllable test cases:
- Sine wave sections with known frequencies
- Abrupt frequency/amplitude changes for tampering simulation
- Silent regions and noise floors
- Known sample rates and bit depths

## Docker Support

The project includes a Dockerfile for containerized deployment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

STREAMLET_SERVER_PORT=8501
STREAMLET_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

To build and run:
```bash
docker build -t audio-forensic-analyzer .
docker run -p 8501:8501 audio-forensic-analyzer
```

Then access the application at `http://localhost:8501`.

## Conclusion

The Audio Forensic Analyzer provides a solid foundation for preliminary audio evidence examination while adhering to important forensic principles. It balances functionality with educational accessibility, making it suitable for learning digital forensics concepts and conducting initial investigations.

Remember: This tool is for preliminary analysis only. Any findings requiring legal or official determination should be examined by qualified digital forensic professionals using validated forensic tools and methodologies.