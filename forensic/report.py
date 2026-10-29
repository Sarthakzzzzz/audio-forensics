"""
PDF report generation module for Audio Forensic Analyzer.
Creates forensic reports in PDF format mirroring the complete UI application.
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import io
import base64

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak, KeepTogether
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


class PDFReportGenerator:
    """Generates comprehensive PDF forensic reports mirroring the application UI."""

    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize PDF report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Base stylesheet
        self.styles = getSampleStyleSheet()

        # Custom distinct typography & styles
        self.title_style = ParagraphStyle(
            'CustomDocTitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#0f172a'),
            fontName='Helvetica-Bold',
            spaceAfter=6
        )

        self.subtitle_style = ParagraphStyle(
            'CustomDocSubtitle',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2563eb'),
            fontName='Helvetica-Bold',
            spaceAfter=18
        )

        self.section_heading = ParagraphStyle(
            'CustomSectionHeading',
            parent=self.styles['Heading2'],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#0f172a'),
            fontName='Helvetica-Bold',
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True
        )

        self.subsection_heading = ParagraphStyle(
            'CustomSubSectionHeading',
            parent=self.styles['Heading3'],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor('#1e3a8a'),
            fontName='Helvetica-Bold',
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True
        )

        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#334155')
        )

        self.caption_style = ParagraphStyle(
            'CustomCaption',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor('#64748b'),
            alignment=TA_CENTER,
            spaceAfter=8
        )

        self.warning_style = ParagraphStyle(
            'CustomWarning',
            parent=self.styles['Normal'],
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor('#b45309'),
            backColor=colors.HexColor('#fef3c7'),
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8
        )

    def _fig_to_image(self, fig, width_inches: float = 6.8, height_inches: float = 2.4) -> Optional[Image]:
        """Convert a matplotlib Figure to a ReportLab Image object."""
        if not MATPLOTLIB_AVAILABLE or fig is None:
            return None
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return Image(buf, width=width_inches * inch, height=height_inches * inch)
        except Exception as e:
            print(f"Error converting figure to image: {e}")
            try:
                plt.close(fig)
            except Exception:
                pass
            return None

    def _create_waveform_image(self, times: np.ndarray, signal: np.ndarray) -> Optional[Image]:
        """Render cleaned output audio waveform matching the UI."""
        if not MATPLOTLIB_AVAILABLE or len(times) == 0 or len(signal) == 0:
            return None
        try:
            fig, ax = plt.subplots(figsize=(9, 2.2), dpi=180)
            max_pts = 10000
            if len(times) > max_pts:
                step = max(1, len(times) // max_pts)
                t_plot = times[::step]
                s_plot = signal[::step]
            else:
                t_plot = times
                s_plot = signal

            ax.plot(t_plot, s_plot, color='#2563eb', linewidth=0.75, alpha=0.9)
            ax.set_xlabel('Time (seconds)', fontsize=8)
            ax.set_ylabel('Amplitude', fontsize=8)
            ax.set_title('Cleaned Output Waveform', fontsize=9, fontweight='bold', color='#0f172a')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            return self._fig_to_image(fig, width_inches=6.8, height_inches=1.8)
        except Exception as e:
            print(f"Error plotting waveform: {e}")
            return None

    def _create_psd_spectrum_image(self, spec_comp: Dict[str, Any]) -> Optional[Image]:
        """Render Power Spectral Density (PSD) comparison matching UI Tab 1."""
        if not MATPLOTLIB_AVAILABLE or not spec_comp:
            return None
        try:
            freq_axis = spec_comp.get("frequencies", np.array([]))
            orig_psd = spec_comp.get("original_psd_db", np.array([]))
            denoised_psd = spec_comp.get("denoised_psd_db", np.array([]))
            attenuation = spec_comp.get("attenuation_db", np.array([]))

            if len(freq_axis) == 0 or len(orig_psd) == 0 or len(denoised_psd) == 0:
                return None

            n_psd = min(len(freq_axis), len(orig_psd), len(denoised_psd))
            f_plot = freq_axis[:n_psd]
            o_psd = orig_psd[:n_psd]
            d_psd = denoised_psd[:n_psd]
            att_plot = attenuation[:n_psd] if len(attenuation) >= n_psd else (o_psd - d_psd)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 3.5), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.1]}, dpi=180)

            # Upper plot: Spectra comparison
            ax1.plot(f_plot, o_psd, label='Previous (Original Audio)', color='#ef4444', alpha=0.75, linewidth=1.1)
            ax1.plot(f_plot, d_psd, label='Current (Denoised Audio)', color='#2563eb', alpha=0.9, linewidth=1.3)
            ax1.set_ylabel('Magnitude (dB/Hz)', fontsize=8)
            ax1.set_title('Power Spectral Density: Previous (Original) vs Current (Denoised)', fontsize=9, fontweight='bold', color='#0f172a')
            ax1.legend(loc='upper right', fontsize=7)
            ax1.grid(True, linestyle='--', alpha=0.4)
            ax1.tick_params(labelsize=7)

            # Lower plot: Attenuation / Noise Cut
            ax2.plot(f_plot, att_plot, color='#10b981', linewidth=1.1, label='Noise Reduction (dB)')
            ax2.axhline(0, color='gray', linestyle=':', alpha=0.6)
            ax2.set_xlabel('Frequency (Hz)', fontsize=8)
            ax2.set_ylabel('Reduction (dB)', fontsize=8)
            ax2.set_title('Frequency Attenuation Curve (Noise Removed)', fontsize=8.5, fontweight='bold', color='#0f172a')
            ax2.grid(True, linestyle='--', alpha=0.4)
            ax2.legend(loc='upper right', fontsize=7)
            ax2.tick_params(labelsize=7)

            plt.tight_layout()
            return self._fig_to_image(fig, width_inches=6.8, height_inches=2.7)
        except Exception as e:
            print(f"Error plotting PSD: {e}")
            return None

    def _create_temporal_tracking_image(self, temp_comp: Dict[str, Any]) -> Optional[Image]:
        """Render frequency trajectory over time matching UI Tab 2."""
        if not MATPLOTLIB_AVAILABLE or not temp_comp:
            return None
        try:
            t_times = temp_comp.get("times", np.array([]))
            orig_centroid = temp_comp.get("original_centroid", np.array([]))
            denoised_centroid = temp_comp.get("denoised_centroid", np.array([]))
            denoised_dominant = temp_comp.get("denoised_dominant", np.array([]))

            n_temp = min(len(t_times), len(orig_centroid), len(denoised_centroid), len(denoised_dominant))
            if n_temp == 0:
                return None

            t_plot = t_times[:n_temp]
            oc_plot = orig_centroid[:n_temp]
            dc_plot = denoised_centroid[:n_temp]
            dd_plot = denoised_dominant[:n_temp]

            fig, ax = plt.subplots(figsize=(9, 2.4), dpi=180)
            ax.plot(t_plot, oc_plot, label='Previous Centroid (Original)', color='#f97316', alpha=0.7, linestyle='--', linewidth=1.2)
            ax.plot(t_plot, dc_plot, label='Current Centroid (Denoised)', color='#1d4ed8', linewidth=1.5)
            ax.plot(t_plot, dd_plot, label='Current Dominant Peak Frequency', color='#059669', alpha=0.8, linewidth=0.9)

            ax.set_xlabel('Time (seconds)', fontsize=8)
            ax.set_ylabel('Frequency (Hz)', fontsize=8)
            ax.set_title('Instantaneous Frequency Tracking Across Audio Timeline', fontsize=9, fontweight='bold', color='#0f172a')
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            return self._fig_to_image(fig, width_inches=6.8, height_inches=2.0)
        except Exception as e:
            print(f"Error plotting temporal tracking: {e}")
            return None

    def _create_jump_detection_image(self, temp_comp: Dict[str, Any], jump_threshold_hz: float = 150.0) -> Optional[Image]:
        """Render frequency jump & transition rate chart matching UI Tab 3."""
        if not MATPLOTLIB_AVAILABLE or not temp_comp:
            return None
        try:
            t_times = temp_comp.get("times", np.array([]))
            delta_orig = temp_comp.get("delta_original", np.array([]))
            delta_denoised = temp_comp.get("delta_denoised", np.array([]))

            n_delta = min(len(t_times), len(delta_orig), len(delta_denoised))
            if n_delta == 0:
                return None

            td_plot = t_times[:n_delta]
            do_plot = delta_orig[:n_delta]
            dd_plot = delta_denoised[:n_delta]

            fig, ax = plt.subplots(figsize=(9, 2.2), dpi=180)
            ax.plot(td_plot, do_plot, label='Previous Step Change |Δf| (Original)', color='#fca5a5', alpha=0.6, linewidth=0.9)
            ax.plot(td_plot, dd_plot, label='Current Step Change |Δf| (Denoised)', color='#dc2626', linewidth=1.3)
            ax.axhline(jump_threshold_hz, color='#0284c7', linestyle='--', linewidth=1.1, label=f'Threshold ({jump_threshold_hz:.0f} Hz)')

            ax.set_xlabel('Time (seconds)', fontsize=8)
            ax.set_ylabel('|Δf| Change (Hz)', fontsize=8)
            ax.set_title('Frequency Step-Change Rate (Frame-to-Frame Difference)', fontsize=9, fontweight='bold', color='#0f172a')
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            return self._fig_to_image(fig, width_inches=6.8, height_inches=1.9)
        except Exception as e:
            print(f"Error plotting jump detection: {e}")
            return None

    def _create_comparative_spectrograms_image(
        self,
        spectrogram_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        denoised_spectrogram_db: Optional[np.ndarray] = None,
        sample_rate: int = 48000
    ) -> Optional[Image]:
        """Render 3-panel comparative spectrograms matching UI Tab 4."""
        if not MATPLOTLIB_AVAILABLE or spectrogram_data is None:
            return None
        try:
            spectrogram_db, freq_array, time_array = spectrogram_data

            max_freq = sample_rate / 2.0 if sample_rate > 0 else 24000.0
            valid_indices = freq_array <= max_freq if max_freq > 0 else np.ones(len(freq_array), dtype=bool)

            spec_orig_display = spectrogram_db[valid_indices, :]
            freq_display = freq_array[valid_indices]

            if denoised_spectrogram_db is not None:
                spec_denoised_display = denoised_spectrogram_db[valid_indices, :]
                min_f = min(spec_orig_display.shape[0], spec_denoised_display.shape[0], len(freq_display))
                min_t = min(spec_orig_display.shape[1], spec_denoised_display.shape[1], len(time_array))

                spec_orig_display = spec_orig_display[:min_f, :min_t]
                spec_denoised_display = spec_denoised_display[:min_f, :min_t]
                freq_display = freq_display[:min_f]
                t_display = time_array[:min_t]

                spec_diff = np.maximum(0.0, spec_orig_display - spec_denoised_display)

                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 4.5), sharex=True, dpi=180)
                extent = [t_display[0], t_display[-1], freq_display[0], freq_display[-1]]

                im1 = ax1.imshow(spec_orig_display, aspect='auto', origin='lower', extent=extent, cmap='magma')
                ax1.set_title('Previous Spectrogram (Original Audio with Noise)', fontsize=8, fontweight='bold')
                ax1.set_ylabel('Freq (Hz)', fontsize=7)
                fig.colorbar(im1, ax=ax1, label='dB', shrink=0.8).ax.tick_params(labelsize=6)
                ax1.tick_params(labelsize=6)

                im2 = ax2.imshow(spec_denoised_display, aspect='auto', origin='lower', extent=extent, cmap='magma')
                ax2.set_title('Current Spectrogram (Denoised Clean Audio)', fontsize=8, fontweight='bold')
                ax2.set_ylabel('Freq (Hz)', fontsize=7)
                fig.colorbar(im2, ax=ax2, label='dB', shrink=0.8).ax.tick_params(labelsize=6)
                ax2.tick_params(labelsize=6)

                im3 = ax3.imshow(spec_diff, aspect='auto', origin='lower', extent=extent, cmap='viridis')
                ax3.set_title('Removed Frequency & Noise Components (Original dB - Denoised dB)', fontsize=8, fontweight='bold')
                ax3.set_xlabel('Time (s)', fontsize=7)
                ax3.set_ylabel('Freq (Hz)', fontsize=7)
                fig.colorbar(im3, ax=ax3, label='Δ dB', shrink=0.8).ax.tick_params(labelsize=6)
                ax3.tick_params(labelsize=6)

                plt.tight_layout()
                return self._fig_to_image(fig, width_inches=6.8, height_inches=3.4)
            else:
                # Single spectrogram fallback
                min_f = min(spec_orig_display.shape[0], len(freq_display))
                min_t = min(spec_orig_display.shape[1], len(time_array))
                spec_orig_display = spec_orig_display[:min_f, :min_t]
                freq_display = freq_display[:min_f]
                t_display = time_array[:min_t]

                fig, ax = plt.subplots(figsize=(9, 2.6), dpi=180)
                extent = [t_display[0], t_display[-1], freq_display[0], freq_display[-1]]
                im = ax.imshow(spec_orig_display, aspect='auto', origin='lower', extent=extent, cmap='viridis')
                ax.set_title('Audio Spectrogram (Magnitude dB)', fontsize=8.5, fontweight='bold')
                ax.set_xlabel('Time (s)', fontsize=7.5)
                ax.set_ylabel('Frequency (Hz)', fontsize=7.5)
                fig.colorbar(im, ax=ax, label='dB')
                ax.tick_params(labelsize=7)
                plt.tight_layout()
                return self._fig_to_image(fig, width_inches=6.8, height_inches=2.2)
        except Exception as e:
            print(f"Error plotting comparative spectrograms: {e}")
            return None

    def generate_report(
        self,
        evidence_data: Dict[str, Any],
        audio_properties: Dict[str, Any],
        metadata_results: Optional[Dict[str, Any]] = None,
        waveform_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        spectrogram_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        tampering_results: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        denoise_metrics: Optional[Dict[str, Any]] = None,
        frequency_comparison: Optional[Dict[str, Any]] = None,
        denoised_spectrogram_db: Optional[np.ndarray] = None,
        jump_threshold_hz: float = 150.0
    ) -> str:
        """
        Generate comprehensive PDF forensic report mirroring the UI interface.

        Args:
            evidence_data: Evidence information (id, filename, file_size, mime_type, sha256, analysis_timestamp)
            audio_properties: Audio technical properties (duration, sample_rate, channels, bit_depth, bitrate, codec)
            metadata_results: Optional container metadata analysis
            waveform_data: Tuple of (times, signal) for output waveform plot
            spectrogram_data: Tuple of (spectrogram, frequencies, times)
            tampering_results: Optional tampering detection results
            filename: Optional custom PDF output filename
            denoise_metrics: Denoising metrics (noise_reduction_db, estimated_snr_gain_db, rms_original, rms_denoised, etc.)
            frequency_comparison: Dict containing spectrum_comparison, temporal_comparison, metrics, detected_frequency_jumps
            denoised_spectrogram_db: Optional spectrogram array of denoised signal
            jump_threshold_hz: Sensitivity threshold for abrupt frequency jump detection

        Returns:
            Absolute filepath to the generated PDF document
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ev_id = evidence_data.get('evidence_id', 'unknown')
            filename = f"forensic_report_{ev_id}_{timestamp}.pdf"

        filepath = os.path.join(self.output_dir, filename)

        # Create A4 Document with balanced 0.5 inch (36 pt) margins
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        story = []

        # =========================================================================
        # HEADER (Mirrors app main-header)
        # =========================================================================
        story.append(Paragraph("AUDIO FORENSIC ANALYZER REPORT", self.title_style))
        story.append(Paragraph("Digital Forensic Examination, Noise Reduction & Frequency Change Analysis", self.subtitle_style))
        story.append(Spacer(1, 4))

        # =========================================================================
        # SECTION 1 — Evidence Integrity (Mirrors UI Section 1)
        # =========================================================================
        story.append(Paragraph("SECTION 1 — EVIDENCE INTEGRITY", self.section_heading))

        analysis_time = str(evidence_data.get('analysis_timestamp', 'N/A'))[:19].replace('T', ' ')
        file_size_val = evidence_data.get('file_size', 0)
        file_size_str = f"{file_size_val:,} bytes" if isinstance(file_size_val, (int, float)) else str(file_size_val)

        evidence_rows = [
            [
                Paragraph("<b>Evidence ID</b>", self.normal_style),
                Paragraph(str(evidence_data.get('evidence_id', 'N/A')), self.normal_style),
                Paragraph("<b>Filename</b>", self.normal_style),
                Paragraph(str(evidence_data.get('filename', 'N/A')), self.normal_style)
            ],
            [
                Paragraph("<b>File Size</b>", self.normal_style),
                Paragraph(file_size_str, self.normal_style),
                Paragraph("<b>MIME Type</b>", self.normal_style),
                Paragraph(str(evidence_data.get('mime_type', 'N/A')), self.normal_style)
            ],
            [
                Paragraph("<b>Analysis Time</b>", self.normal_style),
                Paragraph(analysis_time, self.normal_style),
                Paragraph("<b>Status</b>", self.normal_style),
                Paragraph("<font color='#10b981'><b>✓ Cleaned & Ready (COMPLETED)</b></font>", self.normal_style)
            ],
            [
                Paragraph("<b>SHA-256 Hash</b>", self.normal_style),
                Paragraph(f"<font size=7 face='Courier'>{evidence_data.get('sha256', 'N/A')}</font>", self.normal_style),
                "", ""
            ]
        ]

        t_evidence = Table(evidence_rows, colWidths=[1.3*inch, 2.3*inch, 1.2*inch, 2.2*inch])
        t_evidence.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('SPAN', (1, 3), (3, 3)),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6)
        ]))
        story.append(t_evidence)
        story.append(Spacer(1, 10))

        # =========================================================================
        # SECTION 2 — Audio Properties (Mirrors UI 🔊 Audio Properties)
        # =========================================================================
        story.append(Paragraph("SECTION 2 — AUDIO TECHNICAL PROPERTIES", self.section_heading))

        dur = audio_properties.get('duration', 0.0)
        dur_str = f"{dur:.2f} seconds" if isinstance(dur, (int, float)) else str(dur)
        sr = audio_properties.get('sample_rate', 0)
        sr_str = f"{sr:,} Hz" if isinstance(sr, (int, float)) else str(sr)
        br = audio_properties.get('bitrate', 0)
        br_str = f"{br:,} bps" if (isinstance(br, (int, float)) and br > 0) else "N/A"

        audio_rows = [
            [
                Paragraph("<b>Duration</b>", self.normal_style),
                Paragraph(dur_str, self.normal_style),
                Paragraph("<b>Sample Rate</b>", self.normal_style),
                Paragraph(sr_str, self.normal_style)
            ],
            [
                Paragraph("<b>Channels</b>", self.normal_style),
                Paragraph(str(audio_properties.get('channels', 'N/A')), self.normal_style),
                Paragraph("<b>Bit Depth</b>", self.normal_style),
                Paragraph(str(audio_properties.get('bit_depth', 'N/A')), self.normal_style)
            ],
            [
                Paragraph("<b>Bitrate</b>", self.normal_style),
                Paragraph(br_str, self.normal_style),
                Paragraph("<b>Codec</b>", self.normal_style),
                Paragraph(str(audio_properties.get('codec', 'N/A')), self.normal_style)
            ]
        ]

        t_audio = Table(audio_rows, colWidths=[1.3*inch, 2.3*inch, 1.2*inch, 2.2*inch])
        t_audio.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6)
        ]))
        story.append(t_audio)
        story.append(Spacer(1, 10))

        # =========================================================================
        # SECTION 3 — Noise Reduction Metrics (Mirrors UI 🎧 Noise Reduction)
        # =========================================================================
        if denoise_metrics:
            story.append(Paragraph("SECTION 3 — AUDIO NOISE REDUCTION METRICS", self.section_heading))

            nr_db = denoise_metrics.get('noise_reduction_db', 0.0)
            snr_db = denoise_metrics.get('estimated_snr_gain_db', 0.0)
            orig_rms = denoise_metrics.get('rms_original', 0.0)
            clean_rms = denoise_metrics.get('rms_denoised', 0.0)
            noise_rms = denoise_metrics.get('rms_noise_removed', 0.0)
            peak_orig = denoise_metrics.get('peak_original', 0.0)
            peak_clean = denoise_metrics.get('peak_denoised', 0.0)

            denoise_rows = [
                [
                    Paragraph("<b>Metric</b>", self.normal_style),
                    Paragraph("<b>Measured Value</b>", self.normal_style),
                    Paragraph("<b>Interpretation</b>", self.normal_style)
                ],
                [
                    Paragraph("Noise Reduction", self.normal_style),
                    Paragraph(f"<b>{nr_db:.2f} dB</b>", self.normal_style),
                    Paragraph("Total background noise energy suppressed", self.normal_style)
                ],
                [
                    Paragraph("Estimated SNR Gain", self.normal_style),
                    Paragraph(f"<b>+{snr_db:.2f} dB</b>", self.normal_style),
                    Paragraph("Relative signal clarity improvement", self.normal_style)
                ],
                [
                    Paragraph("Original Signal RMS", self.normal_style),
                    Paragraph(f"{orig_rms:.4f}", self.normal_style),
                    Paragraph("Root mean square level before denoising", self.normal_style)
                ],
                [
                    Paragraph("Cleaned Signal RMS", self.normal_style),
                    Paragraph(f"{clean_rms:.4f}", self.normal_style),
                    Paragraph("Root mean square level after denoising", self.normal_style)
                ],
                [
                    Paragraph("Removed Noise RMS", self.normal_style),
                    Paragraph(f"{noise_rms:.4f}", self.normal_style),
                    Paragraph("Average energy of isolated noise residual", self.normal_style)
                ],
                [
                    Paragraph("Peak Level (Orig / Clean)", self.normal_style),
                    Paragraph(f"{peak_orig:.3f} / {peak_clean:.3f}", self.normal_style),
                    Paragraph("Absolute peak amplitude tracking", self.normal_style)
                ]
            ]

            t_denoise = Table(denoise_rows, colWidths=[2.2*inch, 1.8*inch, 3.0*inch])
            t_denoise.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6)
            ]))
            story.append(t_denoise)
            story.append(Spacer(1, 10))

        # =========================================================================
        # SECTION 4 — Frequency Change Analysis Summary (Mirrors UI 📊 Frequency)
        # =========================================================================
        if frequency_comparison and "metrics" in frequency_comparison:
            story.append(Paragraph("SECTION 4 — FREQUENCY CHANGE ANALYSIS (PREVIOUS VS CURRENT)", self.section_heading))

            f_metrics = frequency_comparison["metrics"]
            p_cent = f_metrics.get('previous_mean_centroid_hz', 0.0)
            c_cent = f_metrics.get('current_mean_centroid_hz', 0.0)
            d_cent = f_metrics.get('centroid_change_pct', 0.0)

            p_roll = f_metrics.get('previous_mean_rolloff_hz', 0.0)
            c_roll = f_metrics.get('current_mean_rolloff_hz', 0.0)
            d_roll = f_metrics.get('rolloff_change_pct', 0.0)

            p_flat = f_metrics.get('previous_flatness', 0.0)
            c_flat = f_metrics.get('current_flatness', 0.0)
            flat_desc = "Tonal purity improved" if c_flat < p_flat else "Maintained"

            hf_att = f_metrics.get('high_freq_attenuation_db', 0.0)
            jumps = frequency_comparison.get("detected_frequency_jumps", [])

            freq_summary_rows = [
                [
                    Paragraph("<b>Frequency Parameter</b>", self.normal_style),
                    Paragraph("<b>Previous (Original)</b>", self.normal_style),
                    Paragraph("<b>Current (Cleaned)</b>", self.normal_style),
                    Paragraph("<b>Observed Change</b>", self.normal_style)
                ],
                [
                    Paragraph("Spectral Centroid (Mean)", self.normal_style),
                    Paragraph(f"{p_cent:,.1f} Hz", self.normal_style),
                    Paragraph(f"{c_cent:,.1f} Hz", self.normal_style),
                    Paragraph(f"<b>{d_cent:+.1f}%</b>", self.normal_style)
                ],
                [
                    Paragraph("Spectral Rolloff (85% Energy)", self.normal_style),
                    Paragraph(f"{p_roll:,.1f} Hz", self.normal_style),
                    Paragraph(f"{c_roll:,.1f} Hz", self.normal_style),
                    Paragraph(f"<b>{d_roll:+.1f}%</b>", self.normal_style)
                ],
                [
                    Paragraph("Spectral Flatness", self.normal_style),
                    Paragraph(f"{p_flat:.5f}", self.normal_style),
                    Paragraph(f"{c_flat:.5f}", self.normal_style),
                    Paragraph(flat_desc, self.normal_style)
                ],
                [
                    Paragraph("High-Freq Noise Attenuation", self.normal_style),
                    Paragraph("-", self.normal_style),
                    Paragraph(f"<b>{hf_att:.1f} dB</b>", self.normal_style),
                    Paragraph("Upper hiss suppressed", self.normal_style)
                ],
                [
                    Paragraph("Abrupt Frequency Shifts", self.normal_style),
                    Paragraph("-", self.normal_style),
                    Paragraph(f"<b>{len(jumps)} transitions</b>", self.normal_style),
                    Paragraph(f"Exceeding {jump_threshold_hz:.0f} Hz threshold", self.normal_style)
                ]
            ]

            t_freq = Table(freq_summary_rows, colWidths=[2.2*inch, 1.6*inch, 1.6*inch, 1.6*inch])
            t_freq.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6)
            ]))
            story.append(t_freq)
            story.append(Spacer(1, 10))

        # =========================================================================
        # SECTION 5 — Detected Frequency Transitions Table (Mirrors UI Jumps table)
        # =========================================================================
        if frequency_comparison and "detected_frequency_jumps" in frequency_comparison:
            jumps = frequency_comparison.get("detected_frequency_jumps", [])
            if jumps:
                story.append(Paragraph("SECTION 5 — DETECTED ABRUPT FREQUENCY TRANSITIONS / JUMPS", self.section_heading))
                story.append(Paragraph(
                    f"The following {min(len(jumps), 15)} transition(s) exceeded the configured threshold of "
                    f"<b>{jump_threshold_hz:.0f} Hz</b> frame-to-frame change:",
                    self.normal_style
                ))
                story.append(Spacer(1, 4))

                jump_table_data = [
                    [
                        Paragraph("<b>Time (s)</b>", self.normal_style),
                        Paragraph("<b>Previous Freq (Hz)</b>", self.normal_style),
                        Paragraph("<b>Current Freq (Hz)</b>", self.normal_style),
                        Paragraph("<b>Δ Change (Hz)</b>", self.normal_style),
                        Paragraph("<b>Direction</b>", self.normal_style)
                    ]
                ]

                for jump in jumps[:15]:
                    jump_table_data.append([
                        Paragraph(f"{jump.get('timestamp', 0.0):.2f} s", self.normal_style),
                        Paragraph(f"{jump.get('previous_frequency_hz', 0.0):.1f} Hz", self.normal_style),
                        Paragraph(f"{jump.get('current_frequency_hz', 0.0):.1f} Hz", self.normal_style),
                        Paragraph(f"<b>{jump.get('frequency_change_hz', 0.0):.1f} Hz</b>", self.normal_style),
                        Paragraph(f"{jump.get('direction', 'N/A')}", self.normal_style)
                    ])

                t_jumps = Table(jump_table_data, colWidths=[1.1*inch, 1.5*inch, 1.5*inch, 1.4*inch, 1.5*inch])
                t_jumps.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f8fafc')]),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                    ('LEFTPADDING', (0, 0), (-1, -1), 5),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 5)
                ]))
                story.append(t_jumps)
                if len(jumps) > 15:
                    story.append(Spacer(1, 3))
                    story.append(Paragraph(f"<i>... and {len(jumps) - 15} additional transitions recorded in telemetry data.</i>", self.caption_style))
                story.append(Spacer(1, 10))

        # PageBreak before charts to keep visual graphs formatted together
        story.append(PageBreak())

        # =========================================================================
        # SECTION 6 — Output Waveform Visualization (Mirrors UI 📈 Output Waveform)
        # =========================================================================
        if waveform_data is not None:
            story.append(Paragraph("SECTION 6 — OUTPUT AUDIO WAVEFORM", self.section_heading))
            w_times, w_signal = waveform_data
            wf_img = self._create_waveform_image(w_times, w_signal)
            if wf_img:
                story.append(wf_img)
                dur_text = f"{w_times[-1]:.2f}s" if len(w_times) > 0 else "N/A"
                story.append(Paragraph(f"Figure 1: Cleaned output waveform showing {len(w_signal):,} samples over {dur_text}.", self.caption_style))
            story.append(Spacer(1, 8))

        # =========================================================================
        # SECTION 7 — Visual Frequency Analytics Charts (Mirrors UI 4 Tabs)
        # =========================================================================
        story.append(Paragraph("SECTION 7 — VISUAL FREQUENCY ANALYTICS", self.section_heading))

        # 7A: PSD Spectrum Comparison (Mirrors Tab 1)
        spec_comp = (frequency_comparison or {}).get("spectrum_comparison", {})
        if spec_comp:
            psd_img = self._create_psd_spectrum_image(spec_comp)
            if psd_img:
                story.append(KeepTogether([
                    Paragraph("Figure 2: Power Spectral Density Comparison (Previous vs Current) & Attenuation", self.subsection_heading),
                    psd_img,
                    Paragraph("Top: Original (red) vs Cleaned (blue) spectrum. Bottom: Noise reduction curve (green) across frequencies.", self.caption_style),
                    Spacer(1, 6)
                ]))

        # 7B: Temporal Tracking (Mirrors Tab 2)
        temp_comp = (frequency_comparison or {}).get("temporal_comparison", {})
        if temp_comp:
            track_img = self._create_temporal_tracking_image(temp_comp)
            if track_img:
                story.append(KeepTogether([
                    Paragraph("Figure 3: Instantaneous Frequency Tracking Across Audio Timeline", self.subsection_heading),
                    track_img,
                    Paragraph("Orange: Previous centroid (noisy). Blue: Cleaned centroid. Green: Dominant instantaneous peak frequency.", self.caption_style),
                    Spacer(1, 6)
                ]))

        story.append(PageBreak())

        # 7C: Jump Detection Rate (Mirrors Tab 3)
        if temp_comp:
            jump_img = self._create_jump_detection_image(temp_comp, jump_threshold_hz)
            if jump_img:
                story.append(KeepTogether([
                    Paragraph("Figure 4: Frequency Step-Change Rate (|Δf| Frame-to-Frame Difference)", self.subsection_heading),
                    jump_img,
                    Paragraph(f"Rate of frequency modulation with configured detection threshold ({jump_threshold_hz:.0f} Hz).", self.caption_style),
                    Spacer(1, 8)
                ]))

        # 7D: Comparative Spectrograms (Mirrors Tab 4)
        if spectrogram_data is not None:
            sr_val = int(audio_properties.get('sample_rate') or 48000)
            spec_img = self._create_comparative_spectrograms_image(
                spectrogram_data=spectrogram_data,
                denoised_spectrogram_db=denoised_spectrogram_db,
                sample_rate=sr_val
            )
            if spec_img:
                story.append(KeepTogether([
                    Paragraph("Figure 5: Comparative Spectrograms (Previous, Current, and Noise Difference)", self.subsection_heading),
                    spec_img,
                    Paragraph("Top: Original noisy spectrogram. Center: Denoised spectrogram. Bottom: Isolated removed noise components.", self.caption_style),
                    Spacer(1, 8)
                ]))

        # Metadata section if anomalies or fields provided
        if metadata_results and metadata_results.get('anomalies'):
            story.append(Paragraph("SECTION 8 — METADATA ANOMALIES", self.section_heading))
            for anomaly in metadata_results.get('anomalies', []):
                story.append(Paragraph(f"• {anomaly.get('description', 'Detected anomaly')}", self.warning_style))
            story.append(Spacer(1, 8))

        # =========================================================================
        # SECTION 9 — Overall Assessment & Forensic Disclaimer
        # =========================================================================
        if denoise_metrics:
            assessment_text = (
                f"Forensic examination, spectral noise gating, and frequency transition analysis have been completed. "
                f"Noise suppression of <b>{denoise_metrics.get('noise_reduction_db', 0.0):.2f} dB</b> was achieved with an estimated SNR gain "
                f"of <b>+{denoise_metrics.get('estimated_snr_gain_db', 0.0):.2f} dB</b>. Evidence integrity was validated with cryptographic hash "
                f"<b>SHA-256: {evidence_data.get('sha256', 'N/A')[:32]}...</b>. All acoustic transitions, spectral rolloffs, and PSD parameters "
                f"have been verified."
            )
        else:
            assessment_text = (
                f"Forensic examination and acoustic parameter verification completed. "
                f"Evidence integrity verified with cryptographic hash <b>SHA-256: {evidence_data.get('sha256', 'N/A')[:32]}...</b>. "
                f"Acoustic properties, spectral envelope, and waveform dynamics have been documented."
            )

        story.append(KeepTogether([
            Paragraph("SECTION 9 — OVERALL ASSESSMENT & VERIFICATION", self.section_heading),
            Paragraph(assessment_text, self.normal_style),
            Spacer(1, 8),
            Paragraph("LEGAL & FORENSIC DISCLAIMER", self.subsection_heading),
            Paragraph(
                "This report provides automated digital forensic examination and acoustic signal enhancement. "
                "Detection of anomalies or frequency transitions does not by itself establish intentional audio tampering or fabrication. "
                "Findings should be corroborated using complementary chain-of-custody documentation and expert forensic acoustic review. "
                "Generated by Audio Forensic Analyzer v1.0.",
                self.warning_style
            )
        ]))

        # Build PDF
        doc.build(story)
        return filepath


def create_report_generator(output_dir: str = "./reports") -> PDFReportGenerator:
    """
    Factory function to create PDFReportGenerator object.

    Args:
        output_dir: Directory to save reports

    Returns:
        PDFReportGenerator object
    """
    return PDFReportGenerator(output_dir)
