"""
Audio Forensic Analyzer - Main Application
A lightweight digital forensic analysis tool for audio evidence.
"""

import streamlit as st
import os
import tempfile
from typing import Optional, Tuple, Dict, Any
import numpy as np

# Import forensic modules
from forensic.evidence import create_evidence, Evidence
from forensic.metadata import analyze_metadata, MetadataAnalyzer
from forensic.waveform import load_audio_file, WaveformAnalyzer
from forensic.spectrogram import create_spectrogram_analyzer, SpectrogramAnalyzer
from forensic.features import create_feature_extractor, FeatureExtractor
from forensic.tampering import create_tampering_detector, TamperingDetector
from forensic.report import create_report_generator, PDFReportGenerator

# Page configuration
st.set_page_config(
    page_title="Audio Forensic Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .section-header {
        background-color: #f8fafc;
        padding: 0.5rem 1rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .section-header h2 {
        color: #1e293b !important;
        margin: 0 !important;
    }
    .status-normal {
        color: #10b981;
        font-weight: bold;
    }
    .status-warning {
        color: #f59e0b;
        font-weight: bold;
    }
    .status-suspicious {
        color: #ef4444;
        font-weight: bold;
    }
    .evidence-info {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bfdbfe;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>AUDIO FORENSIC ANALYZER</h1>
        <p>Digital Forensic Examination of Audio Evidence</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Analysis Settings")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload Audio File",
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
            help="Supported formats: WAV, MP3, FLAC, OGG, M4A"
        )

        # Analysis parameters
        st.subheader("Window Settings")
        window_size = st.slider(
            "Window Size (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
            help="Size of analysis window for anomaly detection"
        )

        overlap = st.slider(
            "Window Overlap",
            min_value=0.0,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Overlap between consecutive windows"
        )

        anomaly_threshold = st.slider(
            "Anomaly Threshold",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Threshold for anomaly detection (standard deviations)"
        )

        use_isolation_forest = st.checkbox(
            "Use Isolation Forest",
            value=False,
            help="Use machine learning for anomaly detection (experimental)"
        )

        if use_isolation_forest:
            contamination = st.slider(
                "Contamination",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="Expected proportion of outliers"
            )
        else:
            contamination = 0.1

    # Main content area
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        try:
            # Initialize analysis components
            with st.spinner("Analyzing evidence..."):
                # Evidence handling
                evidence = create_evidence(temp_file_path)
                evidence_data = evidence.get_evidence_summary()

                # Metadata analysis
                metadata_analyzer = analyze_metadata(temp_file_path)
                metadata_results = metadata_analyzer.get_metadata_summary()

                # Load audio for waveform and feature extraction
                waveform_analyzer = load_audio_file(temp_file_path)
                signal = waveform_analyzer.signal
                sample_rate = waveform_analyzer.sample_rate

                # Audio properties
                audio_properties = waveform_analyzer.get_audio_properties()
                audio_properties.update({
                    'bit_depth': metadata_analyzer.metadata.get('bit_depth'),
                    'bitrate': metadata_analyzer.metadata.get('bitrate'),
                    'codec': metadata_analyzer.metadata.get('codec')
                })

                # Spectrogram generation
                spectrogram_analyzer = create_spectrogram_analyzer(signal, sample_rate)
                spectrogram, frequencies, times = spectrogram_analyzer.generate_spectrogram()
                spectrogram_db = spectrogram_analyzer.get_spectrogram_db()

                # Feature extraction
                feature_extractor = create_feature_extractor(signal, sample_rate)
                features = feature_extractor.extract_all_features()

                # Tampering detection
                tampering_detector = create_tampering_detector(
                    window_size=window_size,
                    overlap=overlap,
                    anomaly_threshold=anomaly_threshold,
                    use_isolation_forest=use_isolation_forest,
                    contamination=contamination
                )

                # Need to approximate hop length for tampering detection
                # Using 512 as default hop length from feature extraction
                tampering_results = tampering_detector.detect_tampering(features, hop_length=512)

                # Get waveform data for display
                waveform_times, waveform_signal = waveform_analyzer.get_waveform_data()

            # Display results
            display_analysis_results(
                evidence_data,
                audio_properties,
                metadata_results,
                (waveform_times, waveform_signal),
                (spectrogram_db, frequencies, times),
                tampering_results
            )

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.error("Please check that the file is a valid audio format and try again.")

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass

    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload an audio file to begin forensic analysis")

        # Show example workflow
        st.subheader("How to Use This Tool")
        st.markdown("""
        1. **Upload an audio file** using the file uploader in the sidebar
        2. **Adjust analysis settings** if needed (window size, overlap, threshold)
        3. **Review the evidence information** including SHA-256 hash for integrity
        4. **Examine metadata** and note any reported anomalies
        5. **Analyze waveform and spectrogram** visualizations
        6. **Review tampering analysis results** for suspicious regions
        7. **Generate a PDF report** for documentation

        ⚠️ **Important Forensic Note**: This tool provides preliminary analysis only.
        Detection of anomalies does not definitively prove manipulation.
        """)

def display_analysis_results(
    evidence_data: Dict[str, Any],
    audio_properties: Dict[str, Any],
    metadata_results: Dict[str, Any],
    waveform_data: Tuple[np.ndarray, np.ndarray],
    spectrogram_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tampering_results: Dict[str, Any]
):
    """Display all analysis results in the main interface."""

    # Evidence Section
    st.markdown('<div class="section-header"><h2>SECTION 1 — Evidence</h2></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Evidence ID", evidence_data['evidence_id'])
        st.metric("Filename", evidence_data['filename'])

    with col2:
        st.metric("File Size", f"{evidence_data['file_size']:,} bytes")
        st.metric("MIME Type", evidence_data['mime_type'])

    with col3:
        st.metric("Analysis Time", evidence_data['analysis_timestamp'][:19].replace('T', ' '))
        # Display SHA-256 prominently
        st.code(evidence_data['sha256'], language=None)

    # Audio Properties Section
    st.markdown('<div class="section-header"><h2>🔊 AUDIO PROPERTIES</h2></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Duration", f"{audio_properties.get('duration', 0):.2f} seconds")
        st.metric("Sample Rate", f"{audio_properties.get('sample_rate', 0):,} Hz")

    with col2:
        st.metric("Channels", audio_properties.get('channels', 'N/A'))
        st.metric("Bit Depth", str(audio_properties.get('bit_depth', 'N/A')))

    with col3:
        bitrate = audio_properties.get('bitrate', 0)
        st.metric("Bitrate", f"{bitrate:,} bps" if bitrate and bitrate > 0 else "N/A")
        st.metric("Codec", audio_properties.get('codec', 'N/A'))

    with col4:
        # Overall status indicator
        suspicious_count = len(tampering_results.get('suspicious_timestamps', []))
        if suspicious_count > 0:
            st.markdown('<p class="status-suspicious">⚠ SUSPICIOUS REGIONS DETECTED</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-normal">✓ NO SIGNIFICANT ANOMALIES DETECTED</p>', unsafe_allow_html=True)

    # Metadata Section
    st.markdown('<div class="section-header"><h2>🏷️ METADATA ANALYSIS</h2></div>', unsafe_allow_html=True)

    metadata = metadata_results.get('metadata', {})
    anomalies = metadata_results.get('anomalies', [])

    if metadata:
        # Display metadata in a nice format
        metadata_items = [(k, v) for k, v in metadata.items() if k not in ['error']]
        if metadata_items:
            # Create two-column layout for metadata
            meta_col1, meta_col2 = st.columns(2)
            mid_point = len(metadata_items) // 2 + len(metadata_items) % 2

            with meta_col1:
                for key, value in metadata_items[:mid_point]:
                    st.text(f"{key}: {value}")

            with meta_col2:
                for key, value in metadata_items[mid_point:]:
                    st.text(f"{key}: {value}")
        else:
            st.info("No metadata found in the audio file.")
    else:
        st.info("No metadata available.")

    # Show metadata anomalies if any
    if anomalies:
        st.warning("⚠️ **Metadata Anomalies Detected:**")
        for anomaly in anomalies:
            st.write(f"• {anomaly.get('description', 'Unknown anomaly')}")

    # Waveform Section
    st.markdown('<div class="section-header"><h2>📈 WAVEFORM VISUALIZATION</h2></div>', unsafe_allow_html=True)

    times, signal = waveform_data

    # Create waveform plot using Streamlit's built-in charting
    import pandas as pd
    waveform_df = pd.DataFrame({
        'Time (s)': times,
        'Amplitude': signal
    })
    st.line_chart(waveform_df.set_index('Time (s)'), height=200)

    st.caption(f"Waveform showing {len(signal):,} samples over {times[-1]:.2f} seconds")

    # Spectrogram Section
    st.markdown('<div class="section-header"><h2>🎨 SPECTROGRAM VISUALIZATION</h2></div>', unsafe_allow_html=True)

    spectrogram_db, freq_array, time_array = spectrogram_data

    # Create a simple spectrogram display using matplotlib if available
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Limit frequency display to maximum frequency of audio (Nyquist frequency)
        max_freq = audio_properties.get('sample_rate', 0) / 2
        if max_freq > 0 and len(freq_array) > 0:
            # Find indices where frequency is within desired range
            valid_freq_indices = freq_array <= max_freq
            if np.any(valid_freq_indices):
                # Truncate data to only include frequencies up to max_freq
                spectrogram_db_limited = spectrogram_db[valid_freq_indices, :]
                freq_array_limited = freq_array[valid_freq_indices]
            else:
                # Fallback to original data if no valid frequencies found
                spectrogram_db_limited = spectrogram_db
                freq_array_limited = freq_array
        else:
            # Fallback to original data if sample rate not available
            spectrogram_db_limited = spectrogram_db
            freq_array_limited = freq_array

        fig, ax = plt.subplots(figsize=(8, 2.5))
        im = ax.imshow(
            spectrogram_db_limited,
            aspect='auto',
            origin='lower',
            extent=[time_array[0], time_array[-1], freq_array_limited[0], freq_array_limited[-1]],
            cmap='viridis'
        )
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram (Magnitude in dB)')
        plt.colorbar(im, ax=ax, label='Magnitude (dB)')
        plt.tight_layout()

        st.pyplot(fig)
        plt.close()
    except ImportError:
        # Fallback if matplotlib not available
        st.info("Spectrogram visualization requires matplotlib. Installing...")
        # Create a simple heatmap using Streamlit
        # Reduce resolution for display
        display_spec = spectrogram_db[::max(1, spectrogram_db.shape[0]//50), ::max(1, spectrogram_db.shape[1]//100)]
        st.image(display_spec, caption="Spectrogram (reduced resolution for display)")

    # Tampering Analysis Section
    st.markdown('<div class="section-header"><h2>🔍 TAMPERING ANALYSIS RESULTS</h2></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Analysis Windows", tampering_results.get('num_windows', 0))

    with col2:
        st.metric("Suspicious Regions", len(tampering_results.get('suspicious_timestamps', [])))

    with col3:
        st.metric("Max Anomaly Score", f"{tampering_results.get('max_anomaly_score', 0):.3f}")

    with col4:
        # Detection method
        method = "Isolation Forest" if st.session_state.get('use_isolation_forest', False) else "Threshold-Based"
        st.metric("Method", method)

    # Show suspicious timestamps
    suspicious_times = tampering_results.get('suspicious_timestamps', [])
    if suspicious_times:
        st.warning("⚠️ **Suspicious Regions Detected:**")
        for i, timestamp in enumerate(suspicious_times[:10]):  # Show first 10
            st.write(f"• Possible audio discontinuity detected at **{timestamp:.2f} seconds**")

        if len(suspicious_times) > 10:
            st.write(f"• ... and {len(suspicious_times) - 10} more suspicious regions")
    else:
        st.success("✓ No significant signal discontinuities detected by this analysis.")

    # Detailed anomalies (if any)
    anomalies = tampering_results.get('anomalies', [])
    if anomalies and st.checkbox("Show detailed anomaly information"):
        st.subheader("Detailed Anomaly Information")
        for i, anomaly in enumerate(anomalies[:5]):  # Show first 5
            with st.expander(f"Anomaly #{i+1} (Score: {anomaly.get('anomaly_score', 0):.3f})"):
                st.json(anomaly)

    # PDF Report Generation Section
    st.markdown('<div class="section-header"><h2>📄 GENERATE FORENSIC REPORT</h2></div>', unsafe_allow_html=True)

    if st.button("📥 Generate PDF Report", type="primary"):
        with st.spinner("Generating PDF report..."):
            try:
                # Create report generator
                report_gen = create_report_generator()

                # Generate report
                report_path = report_gen.generate_report(
                    evidence_data=evidence_data,
                    audio_properties=audio_properties,
                    metadata_results=metadata_results,
                    waveform_data=waveform_data,
                    spectrogram_data=(spectrogram_db, freq_array, time_array),
                    tampering_results=tampering_results
                )

                # Provide download link
                with open(report_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()

                st.success(f"✅ Report generated successfully!")
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=os.path.basename(report_path),
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                st.error("Please ensure reportlab is installed and try again.")

if __name__ == "__main__":
    main()