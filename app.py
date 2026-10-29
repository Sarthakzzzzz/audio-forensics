"""
Audio Forensic Analyzer - Main Application
A lightweight digital forensic analysis tool for audio evidence.
"""

import streamlit as st
import os
import tempfile
import io
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import forensic modules
from forensic.evidence import create_evidence, Evidence
from forensic.metadata import analyze_metadata, MetadataAnalyzer
from forensic.waveform import load_audio_file, WaveformAnalyzer
from forensic.spectrogram import create_spectrogram_analyzer, SpectrogramAnalyzer
from forensic.denoise import create_denoiser, AudioDenoiser
from forensic.frequency import create_frequency_analyzer, FrequencyAnalyzer
from forensic.report import create_report_generator

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
        padding: 1.2rem 0;
        background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .section-header {
        background-color: #f1f5f9;
        padding: 0.5rem 1rem;
        border-left: 5px solid #2563eb;
        border-radius: 4px;
        margin: 1.2rem 0 0.8rem 0;
    }
    .section-header h2 {
        color: #0f172a !important;
        margin: 0 !important;
        font-size: 1.3rem !important;
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
    .audio-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin-bottom: 0.3rem;">AUDIO FORENSIC ANALYZER</h1>
        <p style="margin-top: 0; opacity: 0.9;">Digital Forensic Examination, Noise Reduction & Frequency Change Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Analysis & Processing")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload Audio File",
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'webm', 'opus'],
            help="Supported formats: WAV, MP3, FLAC, OGG, M4A, WEBM, OPUS"
        )

        st.markdown("---")
        st.subheader("🎧 Noise Reduction Settings")

        enable_denoising = st.checkbox(
            "Enable Noise Reduction",
            value=True,
            help="Denoise the audio and compare frequency changes before & after"
        )

        noise_reduction_strength = st.slider(
            "Reduction Strength",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Degree of noise attenuation (0.0 = none, 1.0 = maximum)"
        )

        noise_mode = st.radio(
            "Noise Model",
            ["Stationary (constant hum/hiss)", "Non-Stationary (varying noise)"],
            index=0,
            help="Stationary works best for constant background noise, HVAC, or line hum"
        )

        filter_choice = st.selectbox(
            "Frequency Pre-Filter",
            [
                "None",
                "Bandpass: Speech (80Hz - 4000Hz)",
                "Highpass (>80Hz: remove rumble)",
                "Lowpass (<4000Hz: remove hiss)",
                "Notch 50 Hz (Mains Hum)",
                "Notch 60 Hz (Mains Hum)"
            ],
            index=0,
            help="Apply audio filtering to isolate voice frequencies or remove specific noise"
        )

        jump_threshold_hz = st.slider(
            "Frequency Jump Sensitivity (Hz)",
            min_value=50,
            max_value=600,
            value=150,
            step=25,
            help="Minimum frequency change between frames to flag as an abrupt transition"
        )

    # Main content area
    if uploaded_file is not None:
        # Save uploaded file temporarily
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        try:
            # Initialize analysis components
            with st.spinner("Analyzing evidence, denoising audio, and calculating frequency changes..."):
                # 1. Evidence handling
                evidence = create_evidence(temp_file_path)
                evidence_data = evidence.get_evidence_summary()

                # 2. Metadata analysis
                metadata_analyzer = analyze_metadata(temp_file_path)
                metadata_results = metadata_analyzer.get_metadata_summary()

                # 3. Load audio for waveform and feature extraction
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

                # 4. Spectrogram generation (Original)
                spectrogram_analyzer = create_spectrogram_analyzer(signal, sample_rate)
                spectrogram, frequencies, times = spectrogram_analyzer.generate_spectrogram()
                spectrogram_db = spectrogram_analyzer.get_spectrogram_db()

                # 5. Audio Denoising & Enhancement
                denoiser = create_denoiser(signal, sample_rate)
                if enable_denoising:
                    is_stationary = noise_mode.startswith("Stationary")
                    denoised_signal = denoiser.reduce_noise(
                        prop_decrease=noise_reduction_strength,
                        stationary=is_stationary
                    )

                    # Apply optional filter
                    if filter_choice == "Bandpass: Speech (80Hz - 4000Hz)":
                        denoised_signal = denoiser.apply_filter("bandpass", low_cutoff=80.0, high_cutoff=4000.0)
                    elif filter_choice == "Highpass (>80Hz: remove rumble)":
                        denoised_signal = denoiser.apply_filter("highpass", low_cutoff=80.0)
                    elif filter_choice == "Lowpass (<4000Hz: remove hiss)":
                        denoised_signal = denoiser.apply_filter("lowpass", high_cutoff=4000.0)
                    elif filter_choice == "Notch 50 Hz (Mains Hum)":
                        denoised_signal = denoiser.apply_filter("notch", notch_freq=50.0)
                    elif filter_choice == "Notch 60 Hz (Mains Hum)":
                        denoised_signal = denoiser.apply_filter("notch", notch_freq=60.0)
                else:
                    denoised_signal = signal.copy()

                denoise_metrics = denoiser.calculate_denoising_metrics()
                orig_wav_bytes = AudioDenoiser.to_wav_bytes(signal, sample_rate)
                denoised_wav_bytes = AudioDenoiser.to_wav_bytes(denoised_signal, sample_rate)

                # 9. Frequency Analysis & Comparison (Previous vs Current)
                freq_analyzer = create_frequency_analyzer(sample_rate)
                freq_comparison = freq_analyzer.compare_previous_and_current(
                    original_signal=signal,
                    denoised_signal=denoised_signal,
                    jump_threshold_hz=float(jump_threshold_hz)
                )

                # Spectrogram of denoised signal
                denoised_spec_analyzer = create_spectrogram_analyzer(denoised_signal, sample_rate)
                denoised_spec_analyzer.generate_spectrogram()
                denoised_spectrogram_db = denoised_spec_analyzer.get_spectrogram_db()

            # Display all results
            display_analysis_results(
                evidence_data=evidence_data,
                audio_properties=audio_properties,
                spectrogram_data=(spectrogram_db, frequencies, times),
                orig_audio_bytes=orig_wav_bytes,
                denoised_audio_bytes=denoised_wav_bytes,
                denoise_metrics=denoise_metrics,
                denoised_signal=denoised_signal,
                freq_comparison=freq_comparison,
                denoised_spectrogram_db=denoised_spectrogram_db,
                jump_threshold_hz=float(jump_threshold_hz)
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
        st.info("👆 Please upload an audio file to begin forensic analysis, noise reduction, and frequency change examination")

        st.subheader("Capabilities of This Tool")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🔍 Audio Evidence & Properties
            - **Evidence Integrity**: SHA-256 hash calculation and acquisition identification
            - **Signal Properties**: Sample rate, channels, bit depth, codec, and duration
            - **Output Waveform**: High-resolution amplitude visualization over time
            - **Multi-Format Decoder**: Support for WAV, MP3, FLAC, OGG, M4A, WEBM, OPUS
            """)
        with col2:
            st.markdown("""
            ### 🎧 Audio Enhancement & Frequency Tracking
            - **Noise Removal**: Spectral gating noise reduction (stationary & non-stationary)
            - **Audio Filters**: Speech bandpass, rumble highpass, hiss lowpass, mains hum notch
            - **Cleaned Output Audio**: Immediate in-browser playback and 16-bit WAV download
            - **Frequency Change Analysis**: Power spectrum comparison, frequency drift over time, abrupt jump detection, and comparative spectrograms
            """)


def display_analysis_results(
    evidence_data: Dict[str, Any],
    audio_properties: Dict[str, Any],
    spectrogram_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    orig_audio_bytes: bytes,
    denoised_audio_bytes: bytes,
    denoise_metrics: Dict[str, Any],
    denoised_signal: np.ndarray,
    freq_comparison: Dict[str, Any],
    denoised_spectrogram_db: np.ndarray,
    jump_threshold_hz: float = 150.0
):
    """Display all analysis results in the main interface."""

    # Evidence Section
    st.markdown('<div class="section-header"><h2>SECTION 1 — Evidence Integrity</h2></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Evidence ID", evidence_data['evidence_id'])
        st.metric("Filename", evidence_data['filename'])
    with col2:
        st.metric("File Size", f"{evidence_data['file_size']:,} bytes")
        st.metric("MIME Type", evidence_data['mime_type'])
    with col3:
        st.metric("Analysis Time", evidence_data['analysis_timestamp'][:19].replace('T', ' '))
        st.caption("SHA-256 Hash:")
        st.code(evidence_data['sha256'], language=None)

    # Audio Properties Section
    st.markdown('<div class="section-header"><h2>🔊 Audio Properties</h2></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duration", f"{audio_properties.get('duration', 0):.2f} s")
        st.metric("Sample Rate", f"{audio_properties.get('sample_rate', 0):,} Hz")
    with col2:
        st.metric("Channels", audio_properties.get('channels', 'N/A'))
        st.metric("Bit Depth", str(audio_properties.get('bit_depth', 'N/A')))
    with col3:
        bitrate = audio_properties.get('bitrate', 0)
        st.metric("Bitrate", f"{bitrate:,} bps" if bitrate and bitrate > 0 else "N/A")
        st.metric("Codec", audio_properties.get('codec', 'N/A'))
    with col4:
        st.metric("Status", "Cleaned & Ready")
        st.markdown('<p class="status-normal">✓ COMPLETED</p>', unsafe_allow_html=True)

    # =========================================================================
    # AUDIO DENOISING & OUTPUT PLAYBACK
    # =========================================================================
    st.markdown('<div class="section-header"><h2>🎧 Audio Noise Reduction & Output Audio</h2></div>', unsafe_allow_html=True)

    # Metrics row for Denoising
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Noise Reduction", f"{denoise_metrics.get('noise_reduction_db', 0.0):.2f} dB")
    with m_col2:
        st.metric("Estimated SNR Gain", f"{denoise_metrics.get('estimated_snr_gain_db', 0.0):.2f} dB")
    with m_col3:
        st.metric("Original Signal RMS", f"{denoise_metrics.get('rms_original', 0.0):.4f}")
    with m_col4:
        st.metric("Cleaned Signal RMS", f"{denoise_metrics.get('rms_denoised', 0.0):.4f}")

    # Output Audio Player & Download
    st.markdown("""
    <div class="audio-card">
        <h4 style="margin-top: 0; color: #1e3a8a;">🎧 Cleaned Output Audio</h4>
    </div>
    """, unsafe_allow_html=True)

    player_col, dl_col = st.columns([3, 1])
    with player_col:
        st.audio(denoised_audio_bytes, format="audio/wav")
    with dl_col:
        st.download_button(
            label="📥 Download Cleaned Audio (.wav)",
            data=denoised_audio_bytes,
            file_name=f"cleaned_{os.path.splitext(evidence_data['filename'])[0]}.wav",
            mime="audio/wav",
            help="Download the noise-reduced audio file",
            type="primary",
            use_container_width=True
        )

    # =========================================================================
    # FREQUENCY CHANGE ANALYSIS (PREVIOUS VS CURRENT)
    # =========================================================================
    st.markdown('<div class="section-header"><h2>📊 Frequency Change Analysis (Previous vs Current)</h2></div>', unsafe_allow_html=True)

    f_metrics = freq_comparison.get("metrics", {})
    f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)

    with f_col1:
        st.metric(
            "Previous Mean Frequency",
            f"{f_metrics.get('previous_mean_centroid_hz', 0.0):,.1f} Hz",
            help="Original mean spectral centroid"
        )
    with f_col2:
        st.metric(
            "Current Mean Frequency",
            f"{f_metrics.get('current_mean_centroid_hz', 0.0):,.1f} Hz",
            delta=f"{f_metrics.get('centroid_change_pct', 0.0):+.1f}%",
            help="Cleaned mean spectral centroid and percentage change"
        )
    with f_col3:
        st.metric(
            "Rolloff (85% Energy)",
            f"{f_metrics.get('current_mean_rolloff_hz', 0.0):,.1f} Hz",
            delta=f"{f_metrics.get('rolloff_change_pct', 0.0):+.1f}%",
            help="Spectral rolloff before vs after denoising"
        )
    with f_col4:
        st.metric(
            "High-Freq Attenuation",
            f"{f_metrics.get('high_freq_attenuation_db', 0.0):.1f} dB",
            help="Amount of high-frequency noise suppressed"
        )
    with f_col5:
        jump_count = len(freq_comparison.get("detected_frequency_jumps", []))
        st.metric(
            "Abrupt Freq Shifts",
            f"{jump_count}",
            help="Number of sharp frequency transitions detected across time"
        )

    # Tabbed Frequency Visualizations
    tab_spectrum, tab_temporal, tab_jumps, tab_spectrograms = st.tabs([
        "📈 Frequency Spectrum (PSD) Comparison",
        "⏱️ Frequency Tracking Over Time",
        "⚡ Frequency Change / Jump Detection",
        "🎨 Comparative Spectrograms"
    ])

    spec_comp = freq_comparison.get("spectrum_comparison", {})
    freq_axis = spec_comp.get("frequencies", np.array([]))
    orig_psd = spec_comp.get("original_psd_db", np.array([]))
    denoised_psd = spec_comp.get("denoised_psd_db", np.array([]))
    attenuation = spec_comp.get("attenuation_db", np.array([]))

    with tab_spectrum:
        st.subheader("Previous vs Current Frequency Spectrum (Power Spectral Density)")
        st.caption("Compares frequency magnitude before (previous) and after (current) denoising. The green curve shows the noise reduction (attenuation) across all frequencies.")

        if len(freq_axis) > 0 and len(orig_psd) > 0 and len(denoised_psd) > 0:
            n_psd = min(len(freq_axis), len(orig_psd), len(denoised_psd))
            f_plot = freq_axis[:n_psd]
            o_psd = orig_psd[:n_psd]
            d_psd = denoised_psd[:n_psd]
            att_plot = attenuation[:n_psd] if len(attenuation) >= n_psd else (o_psd - d_psd)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={'height_ratios': [2.5, 1.2]})

            # Upper plot: Spectra comparison
            ax1.plot(f_plot, o_psd, label='Previous (Original Audio)', color='#ef4444', alpha=0.75, linewidth=1.2)
            ax1.plot(f_plot, d_psd, label='Current (Denoised Audio)', color='#2563eb', alpha=0.9, linewidth=1.4)
            ax1.set_ylabel('Magnitude (dB/Hz)')
            ax1.set_title('Power Spectral Density: Previous (Original) vs Current (Denoised)')
            ax1.legend(loc='upper right')
            ax1.grid(True, linestyle='--', alpha=0.5)

            # Lower plot: Attenuation / Noise Cut
            ax2.plot(f_plot, att_plot, color='#10b981', linewidth=1.2, label='Noise Reduction (dB)')
            ax2.axhline(0, color='gray', linestyle=':', alpha=0.6)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Reduction (dB)')
            ax2.set_title('Frequency Attenuation Curve (Noise Removed)')
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.legend(loc='upper right')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    temp_comp = freq_comparison.get("temporal_comparison", {})
    t_times = temp_comp.get("times", np.array([]))
    orig_centroid = temp_comp.get("original_centroid", np.array([]))
    denoised_centroid = temp_comp.get("denoised_centroid", np.array([]))
    orig_dominant = temp_comp.get("original_dominant", np.array([]))
    denoised_dominant = temp_comp.get("denoised_dominant", np.array([]))

    with tab_temporal:
        st.subheader("Frequency Dynamics Over Time: Previous vs Current")
        st.caption("Tracks how frequency shifts across the timeline, comparing the noisy original frequency curve with the cleaned frequency trajectory.")

        n_temp = min(len(t_times), len(orig_centroid), len(denoised_centroid), len(denoised_dominant))
        if n_temp > 0:
            t_plot = t_times[:n_temp]
            oc_plot = orig_centroid[:n_temp]
            dc_plot = denoised_centroid[:n_temp]
            dd_plot = denoised_dominant[:n_temp]

            fig, ax = plt.subplots(figsize=(10, 3.8))
            ax.plot(t_plot, oc_plot, label='Previous Spectral Centroid (Original)', color='#f97316', alpha=0.7, linestyle='--', linewidth=1.3)
            ax.plot(t_plot, dc_plot, label='Current Spectral Centroid (Denoised)', color='#1d4ed8', linewidth=1.8)
            ax.plot(t_plot, dd_plot, label='Current Dominant Peak Frequency', color='#059669', alpha=0.8, linewidth=1.0)

            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title('Instantaneous Frequency Tracking Across Audio Timeline')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab_jumps:
        st.subheader("Abrupt Frequency Change & Transition Detector")
        st.caption("Identifies sudden jumps and shifts in frequency between adjacent frames, useful for finding splice cuts, edits, or frequency modulation.")

        delta_orig = temp_comp.get("delta_original", np.array([]))
        delta_denoised = temp_comp.get("delta_denoised", np.array([]))

        n_delta = min(len(t_times), len(delta_orig), len(delta_denoised))
        if n_delta > 0:
            td_plot = t_times[:n_delta]
            do_plot = delta_orig[:n_delta]
            dd_plot = delta_denoised[:n_delta]

            fig, ax = plt.subplots(figsize=(10, 2.8))
            ax.plot(td_plot, do_plot, label='Previous Step Change |Δf| (Original)', color='#fca5a5', alpha=0.6, linewidth=1.0)
            ax.plot(td_plot, dd_plot, label='Current Step Change |Δf| (Denoised)', color='#dc2626', linewidth=1.5)
            ax.axhline(jump_threshold_hz, color='#0284c7', linestyle='--', label=f'Threshold ({jump_threshold_hz} Hz)')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('|Δf| Change (Hz)')
            ax.set_title('Frequency Step-Change Rate (Frame-to-Frame Difference)')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        jumps = freq_comparison.get("detected_frequency_jumps", [])
        if jumps:
            st.markdown(f"**Detected {len(jumps)} Significant Frequency Transition(s):**")
            jump_df = pd.DataFrame(jumps)
            st.dataframe(
                jump_df[['timestamp', 'previous_frequency_hz', 'current_frequency_hz', 'frequency_change_hz', 'direction']],
                column_config={
                    "timestamp": st.column_config.NumberColumn("Time (s)", format="%.2f s"),
                    "previous_frequency_hz": st.column_config.NumberColumn("Previous Freq (Hz)", format="%.1f Hz"),
                    "current_frequency_hz": st.column_config.NumberColumn("Current Freq (Hz)", format="%.1f Hz"),
                    "frequency_change_hz": st.column_config.NumberColumn("Δ Change (Hz)", format="%.1f Hz"),
                    "direction": st.column_config.TextColumn("Direction")
                },
                use_container_width=True
            )
        else:
            st.success("✓ No abrupt frequency jumps exceeded the detection threshold.")

    spectrogram_db, freq_array, time_array = spectrogram_data

    with tab_spectrograms:
        st.subheader("Visual Spectrogram Comparison")
        st.caption("Side-by-side view of Original vs Cleaned audio and the exact Removed Noise spectrogram.")

        max_freq = audio_properties.get('sample_rate', 0) / 2
        valid_indices = freq_array <= max_freq if max_freq > 0 else np.ones(len(freq_array), dtype=bool)

        spec_orig_display = spectrogram_db[valid_indices, :]
        spec_denoised_display = denoised_spectrogram_db[valid_indices, :]
        freq_display = freq_array[valid_indices]

        # Ensure exact matching dimensions
        min_f = min(spec_orig_display.shape[0], spec_denoised_display.shape[0], len(freq_display))
        min_t = min(spec_orig_display.shape[1], spec_denoised_display.shape[1], len(time_array))

        spec_orig_display = spec_orig_display[:min_f, :min_t]
        spec_denoised_display = spec_denoised_display[:min_f, :min_t]
        freq_display = freq_display[:min_f]
        t_display = time_array[:min_t]

        # Calculate difference (Removed noise)
        spec_diff = np.maximum(0.0, spec_orig_display - spec_denoised_display)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)

        extent = [t_display[0], t_display[-1], freq_display[0], freq_display[-1]]

        im1 = ax1.imshow(spec_orig_display, aspect='auto', origin='lower', extent=extent, cmap='magma')
        ax1.set_title('Previous Spectrogram (Original Audio with Noise)')
        ax1.set_ylabel('Freq (Hz)')
        fig.colorbar(im1, ax=ax1, label='dB')

        im2 = ax2.imshow(spec_denoised_display, aspect='auto', origin='lower', extent=extent, cmap='magma')
        ax2.set_title('Current Spectrogram (Denoised Clean Audio)')
        ax2.set_ylabel('Freq (Hz)')
        fig.colorbar(im2, ax=ax2, label='dB')

        im3 = ax3.imshow(spec_diff, aspect='auto', origin='lower', extent=extent, cmap='viridis')
        ax3.set_title('Removed Frequency & Noise Components (Original dB - Denoised dB)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Freq (Hz)')
        fig.colorbar(im3, ax=ax3, label='Δ dB')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # =========================================================================
    # WAVEFORM SECTION
    # =========================================================================
    st.markdown('<div class="section-header"><h2>📈 Output Audio Waveform</h2></div>', unsafe_allow_html=True)

    sr = int(audio_properties.get('sample_rate') or 48000)
    w_times = np.arange(len(denoised_signal), dtype=np.float64) / sr
    max_pts = 15000
    if len(w_times) > max_pts:
        step = max(1, len(w_times) // max_pts)
        w_times = w_times[::step]
        w_signal = denoised_signal[::step]
    else:
        w_signal = denoised_signal

    min_len = min(len(w_times), len(w_signal))
    w_times = w_times[:min_len]
    w_signal = w_signal[:min_len]

    waveform_df = pd.DataFrame({
        'Time (s)': w_times,
        'Amplitude': w_signal
    })
    st.line_chart(waveform_df.set_index('Time (s)'), height=220)
    duration_val = w_times[-1] if len(w_times) > 0 else 0.0
    st.caption(f"Cleaned output waveform showing {len(denoised_signal):,} samples over {duration_val:.2f} seconds")

    # =========================================================================
    # PDF REPORT OPTION
    # =========================================================================
    st.markdown('<div class="section-header"><h2>📄 Export PDF Forensic & Audio Report</h2></div>', unsafe_allow_html=True)

    rep_col1, rep_col2 = st.columns([2.5, 1.2])
    with rep_col1:
        st.write("Generate a printable PDF report documenting evidence verification hashes (SHA-256), audio technical properties, embedded waveform & spectrogram plots, noise reduction statistics, and before/after frequency comparison metrics.")

    with rep_col2:
        if st.button("📥 Generate PDF Report", type="primary", use_container_width=True):
            with st.spinner("Compiling PDF report..."):
                try:
                    report_gen = create_report_generator()
                    w_times_report = np.arange(len(denoised_signal), dtype=np.float64) / sr
                    report_path = report_gen.generate_report(
                        evidence_data=evidence_data,
                        audio_properties=audio_properties,
                        waveform_data=(w_times_report, denoised_signal),
                        spectrogram_data=spectrogram_data,
                        denoise_metrics=denoise_metrics,
                        frequency_comparison=freq_comparison
                    )

                    with open(report_path, "rb") as pdf_file:
                        st.session_state['report_pdf_bytes'] = pdf_file.read()
                        st.session_state['report_pdf_name'] = os.path.basename(report_path)

                    st.success("✅ PDF report compiled successfully!")
                except Exception as e:
                    st.error(f"Error generating PDF report: {str(e)}")

        if 'report_pdf_bytes' in st.session_state and st.session_state.get('report_pdf_name'):
            st.download_button(
                label=f"⬇️ Download {st.session_state['report_pdf_name']}",
                data=st.session_state['report_pdf_bytes'],
                file_name=st.session_state['report_pdf_name'],
                mime="application/pdf",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
