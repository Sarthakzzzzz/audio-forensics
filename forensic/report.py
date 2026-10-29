"""
PDF report generation module for Audio Forensic Analyzer.
Creates forensic reports in PDF format.
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import io
import base64

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


class PDFReportGenerator:
    """Generates PDF forensic reports."""

    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize PDF report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Define styles
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        self.normal_style = self.styles['Normal']
        self.warning_style = ParagraphStyle(
            'Warning',
            parent=self.styles['Normal'],
            textColor=colors.orange,
            backColor=colors.lightyellow
        )

    def _create_waveform_image(
        self,
        times: np.ndarray,
        signal: np.ndarray,
        width: int = 800,
        height: int = 200
    ) -> str:
        """
        Create waveform image for report.

        Args:
            times: Time array
            signal: Signal array
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Base64 encoded image string
        """
        if not MATPLOTLIB_AVAILABLE:
            return ""

        try:
            plt.figure(figsize=(width/100, height/100), dpi=100)
            plt.plot(times, signal, 'b-', linewidth=0.5)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Waveform')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close()
            return img_str
        except Exception as e:
            print(f"Error creating waveform image: {e}")
            return ""

    def _create_spectrogram_image(
        self,
        spectrogram: np.ndarray,
        frequencies: np.ndarray,
        times: np.ndarray,
        width: int = 800,
        height: int = 400
    ) -> str:
        """
        Create spectrogram image for report.

        Args:
            spectrogram: Spectrogram array
            frequencies: Frequency array
            times: Time array
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Base64 encoded image string
        """
        if not MATPLOTLIB_AVAILABLE:
            return ""

        try:
            plt.figure(figsize=(width/100, height/100), dpi=100)
            plt.imshow(
                20 * np.log10(np.maximum(spectrogram, 1e-10)),
                aspect='auto',
                origin='lower',
                extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
                cmap='viridis'
            )
            plt.colorbar(label='Magnitude (dB)')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Spectrogram')
            plt.tight_layout()

            # Save to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close()
            return img_str
        except Exception as e:
            print(f"Error creating spectrogram image: {e}")
            return ""

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
        frequency_comparison: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate PDF forensic report.

        Args:
            evidence_data: Evidence information
            audio_properties: Audio properties
            metadata_results: Metadata analysis results
            waveform_data: Tuple of (times, signal) for waveform
            spectrogram_data: Tuple of (spectrogram, frequencies, times) for spectrogram
            tampering_results: Tampering detection results
            filename: Optional custom filename

        Returns:
            Path to generated PDF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forensic_report_{evidence_data.get('evidence_id', 'unknown')}_{timestamp}.pdf"

        filepath = os.path.join(self.output_dir, filename)

        # Create document
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        story = []

        # Title
        title = Paragraph("AUDIO FORENSIC ANALYZER REPORT", self.title_style)
        story.append(title)
        story.append(Spacer(1, 20))

        # Subtitle
        subtitle = Paragraph("Digital Forensic Examination of Audio Evidence", self.styles['Heading2'])
        story.append(subtitle)
        story.append(Spacer(1, 30))

        # Evidence Information
        story.append(Paragraph("EVIDENCE INFORMATION", self.heading_style))

        evidence_table_data = [
            ["Field", "Value"],
            ["Evidence ID", evidence_data.get('evidence_id', 'N/A')],
            ["Filename", evidence_data.get('filename', 'N/A')],
            ["Analysis Timestamp", evidence_data.get('analysis_timestamp', 'N/A')],
            ["File Size", f"{evidence_data.get('file_size', 0)} bytes"],
            ["MIME Type", evidence_data.get('mime_type', 'N/A')],
            ["SHA-256 Hash", evidence_data.get('sha256', 'N/A')]
        ]

        evidence_table = Table(evidence_table_data, colWidths=[2*inch, 4*inch])
        evidence_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(evidence_table)
        story.append(Spacer(1, 20))

        # Audio Properties
        story.append(Paragraph("AUDIO PROPERTIES", self.heading_style))

        audio_table_data = [
            ["Property", "Value"],
            ["Duration", f"{audio_properties.get('duration', 0):.2f} seconds"],
            ["Sample Rate", f"{audio_properties.get('sample_rate', 0)} Hz"],
            ["Channels", str(audio_properties.get('channels', 'N/A'))],
            ["Bit Depth", str(audio_properties.get('bit_depth', 'N/A'))],
            ["Bitrate", f"{audio_properties.get('bitrate', 0)} bps" if audio_properties.get('bitrate') else "N/A"],
            ["Codec", audio_properties.get('codec', 'N/A')]
        ]

        audio_table = Table(audio_table_data, colWidths=[2*inch, 4*inch])
        audio_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(audio_table)
        story.append(Spacer(1, 20))

        # Metadata (optional if provided)
        if metadata_results and metadata_results.get('metadata'):
            story.append(Paragraph("METADATA ANALYSIS", self.heading_style))
            metadata = metadata_results.get('metadata', {})
            metadata_table_data = [["Field", "Value"]]
            for key, value in metadata.items():
                if key not in ['error']:  # Skip error field for cleaner display
                    metadata_table_data.append([str(key), str(value)])

            if len(metadata_table_data) > 1:  # Only add table if we have data
                metadata_table = Table(metadata_table_data, colWidths=[2*inch, 4*inch])
                metadata_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(metadata_table)

            anomalies = metadata_results.get('anomalies', [])
            if anomalies:
                story.append(Spacer(1, 10))
                story.append(Paragraph("METADATA ANOMALIES DETECTED:", self.warning_style))
                for anomaly in anomalies:
                    anomaly_text = f"• {anomaly.get('description', 'Unknown anomaly')}"
                    story.append(Paragraph(anomaly_text, self.warning_style))
            story.append(Spacer(1, 20))

        # Waveform Image
        if waveform_data is not None:
            story.append(Paragraph("WAVEFORM VISUALIZATION", self.heading_style))
            times, signal = waveform_data
            waveform_img_b64 = self._create_waveform_image(times, signal)
            if waveform_img_b64 and MATPLOTLIB_AVAILABLE:
                import base64
                from reportlab.platypus import Image
                img_data = base64.b64decode(waveform_img_b64)
                img_buf = io.BytesIO(img_data)
                img = Image(img_buf, width=6*inch, height=1.5*inch)
                story.append(img)
            else:
                story.append(Paragraph("Waveform visualization not available.", self.normal_style))
            story.append(Spacer(1, 20))

        # Spectrogram Image
        if spectrogram_data is not None:
            story.append(Paragraph("SPECTROGRAM VISUALIZATION", self.heading_style))
            spectrogram, frequencies, times_spec = spectrogram_data
            spectrogram_img_b64 = self._create_spectrogram_image(spectrogram, frequencies, times_spec)
            if spectrogram_img_b64 and MATPLOTLIB_AVAILABLE:
                import base64
                from reportlab.platypus import Image
                img_data = base64.b64decode(spectrogram_img_b64)
                img_buf = io.BytesIO(img_data)
                img = Image(img_buf, width=6*inch, height=3*inch)
                story.append(img)
            else:
                story.append(Paragraph("Spectrogram visualization not available.", self.normal_style))
            story.append(Spacer(1, 20))

        # Tampering Analysis Results (only if tampering detection was executed)
        if tampering_results and tampering_results.get('num_windows', 0) > 0:
            story.append(Paragraph("TAMPERING ANALYSIS RESULTS", self.heading_style))

            tampering_table_data = [
                ["Metric", "Value"],
                ["Number of Windows", str(tampering_results.get('num_windows', 0))],
                ["Number of Suspicious Regions", str(len(tampering_results.get('suspicious_timestamps', [])))],
                ["Maximum Anomaly Score", f"{tampering_results.get('max_anomaly_score', 0):.3f}"]
            ]

            tampering_table = Table(tampering_table_data, colWidths=[3*inch, 3*inch])
            tampering_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(tampering_table)
            story.append(Spacer(1, 10))

            suspicious_times = tampering_results.get('suspicious_timestamps', [])
            if suspicious_times:
                story.append(Paragraph("SUSPICIOUS TIMESTAMPS:", self.warning_style))
                for i, timestamp in enumerate(suspicious_times[:10]):  # Limit to first 10
                    story.append(Paragraph(f"• Possible audio discontinuity detected at {timestamp:.2f} seconds.", self.warning_style))
                if len(suspicious_times) > 10:
                    story.append(Paragraph(f"• ... and {len(suspicious_times) - 10} more suspicious regions.", self.warning_style))
            else:
                story.append(Paragraph("No significant signal discontinuities detected.", self.normal_style))
            story.append(Spacer(1, 20))

        # Audio Enhancement & Frequency Analysis Section (if available)
        if denoise_metrics or frequency_comparison:
            story.append(Paragraph("NOISE REDUCTION & FREQUENCY ANALYSIS", self.heading_style))

            if denoise_metrics:
                enhancement_data = [
                    ["Denoising Metric", "Value"],
                    ["Noise Reduction (dB)", f"{denoise_metrics.get('noise_reduction_db', 0.0):.2f} dB"],
                    ["Estimated SNR Gain", f"{denoise_metrics.get('estimated_snr_gain_db', 0.0):.2f} dB"],
                    ["Original RMS", f"{denoise_metrics.get('rms_original', 0.0):.4f}"],
                    ["Denoised RMS", f"{denoise_metrics.get('rms_denoised', 0.0):.4f}"],
                    ["Removed Noise RMS", f"{denoise_metrics.get('rms_noise_removed', 0.0):.4f}"]
                ]
                enh_table = Table(enhancement_data, colWidths=[3*inch, 3*inch])
                enh_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(enh_table)
                story.append(Spacer(1, 15))

            if frequency_comparison and "metrics" in frequency_comparison:
                f_metrics = frequency_comparison["metrics"]
                freq_data = [
                    ["Frequency Metric", "Previous (Original)", "Current (Denoised)", "Change"],
                    [
                        "Spectral Centroid",
                        f"{f_metrics.get('previous_mean_centroid_hz', 0.0)} Hz",
                        f"{f_metrics.get('current_mean_centroid_hz', 0.0)} Hz",
                        f"{f_metrics.get('centroid_change_pct', 0.0):+.1f}%"
                    ],
                    [
                        "Spectral Rolloff (85%)",
                        f"{f_metrics.get('previous_mean_rolloff_hz', 0.0)} Hz",
                        f"{f_metrics.get('current_mean_rolloff_hz', 0.0)} Hz",
                        f"{f_metrics.get('rolloff_change_pct', 0.0):+.1f}%"
                    ],
                    [
                        "Spectral Flatness",
                        f"{f_metrics.get('previous_flatness', 0.0):.5f}",
                        f"{f_metrics.get('current_flatness', 0.0):.5f}",
                        "Tonal improvement" if f_metrics.get('current_flatness', 0) < f_metrics.get('previous_flatness', 0) else "N/A"
                    ],
                    [
                        "High Freq Attenuation",
                        "-",
                        f"{f_metrics.get('high_freq_attenuation_db', 0.0):.1f} dB",
                        "Noise suppressed"
                    ]
                ]
                freq_table = Table(freq_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1*inch])
                freq_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(freq_table)
                story.append(Spacer(1, 15))

            # Detected frequency jumps
            jumps = frequency_comparison.get("detected_frequency_jumps", []) if frequency_comparison else []
            if jumps:
                story.append(Paragraph("DETECTED FREQUENCY TRANSITIONS / JUMPS:", self.warning_style))
                for jump in jumps[:5]:
                    story.append(Paragraph(
                        f"• At {jump['timestamp']:.2f}s: frequency changed from {jump['previous_frequency_hz']} Hz to "
                        f"{jump['current_frequency_hz']} Hz (Δ = {jump['frequency_change_hz']} Hz, {jump['direction']})",
                        self.normal_style
                    ))
                story.append(Spacer(1, 15))

        # Overall Assessment
        story.append(Paragraph("OVERALL ASSESSMENT", self.heading_style))

        suspicious_times = (tampering_results or {}).get('suspicious_timestamps', [])

        if denoise_metrics and denoise_metrics.get('noise_reduction_db', 0) > 0:
            red_db = denoise_metrics.get('noise_reduction_db', 0.0)
            snr_gain = denoise_metrics.get('estimated_snr_gain_db', 0.0)
            assessment_text = (
                f"Audio noise reduction and frequency analysis completed successfully. "
                f"Achieved {red_db:.2f} dB noise reduction with an estimated SNR improvement of {snr_gain:.2f} dB. "
                f"Acoustic properties, spectral envelope, and waveform dynamics have been analyzed and documented."
            )
        elif suspicious_times:
            assessment_text = (
                "Potential anomaly detected. Further forensic examination is recommended. "
                "The detection of signal discontinuities suggests possible editing or tampering, "
                "but additional analysis is required to confirm any manipulation."
            )
        else:
            assessment_text = (
                "Audio signal processed and analyzed. Technical parameters, acoustic properties, and "
                "waveform metrics have been documented in this report."
            )

        story.append(Paragraph(assessment_text, self.normal_style))
        story.append(Spacer(1, 30))

        # Forensic Disclaimer
        story.append(Paragraph("FORENSIC DISCLAIMER", self.heading_style))
        disclaimer_text = (
            "This tool provides preliminary digital forensic analysis. Detection of an anomaly "
            "does not by itself establish that an audio recording was manipulated. Findings should "
            "be corroborated using additional forensic methods and appropriate reference evidence. "
            "This analysis is intended for educational and preliminary investigative purposes only."
        )
        story.append(Paragraph(disclaimer_text, self.warning_style))

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