"""
Evidence handling module for Audio Forensic Analyzer.
Handles file information, hashing, and evidence identification.
"""

import os
import hashlib
import mimetypes
from datetime import datetime
from typing import Dict, Any, Optional
import magic


class Evidence:
    """Handles evidence acquisition and integrity verification."""

    def __init__(self, file_path: str):
        """
        Initialize evidence handler.

        Args:
            file_path: Path to the audio file
        """
        self.file_path = file_path
        self.evidence_id = self._generate_evidence_id()
        self.file_info = self._get_file_info()
        self.sha256_hash = self._calculate_sha256()

    def _generate_evidence_id(self) -> str:
        """Generate a unique evidence ID based on timestamp."""
        return f"EVID_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def _get_file_info(self) -> Dict[str, Any]:
        """Extract basic file information."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        stat = os.stat(self.file_path)
        file_size = stat.st_size

        # Get MIME type using python-magic
        mime_type = magic.from_file(self.file_path, mime=True)

        # Fallback to mimetypes if magic fails
        if not mime_type or mime_type == 'application/octet-stream':
            mime_type, _ = mimetypes.guess_type(self.file_path)
            if not mime_type:
                mime_type = 'unknown/unknown'

        return {
            'filename': os.path.basename(self.file_path),
            'file_size': file_size,
            'mime_type': mime_type,
            'full_path': self.file_path
        }

    def _calculate_sha256(self) -> str:
        """Calculate SHA-256 hash of the file."""
        sha256_hash = hashlib.sha256()

        with open(self.file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def get_evidence_summary(self) -> Dict[str, Any]:
        """Get a summary of evidence information."""
        return {
            'evidence_id': self.evidence_id,
            'filename': self.file_info['filename'],
            'file_size': self.file_info['file_size'],
            'mime_type': self.file_info['mime_type'],
            'sha256': self.sha256_hash,
            'analysis_timestamp': datetime.now().isoformat()
        }


def create_evidence(file_path: str) -> Evidence:
    """
    Factory function to create Evidence object.

    Args:
        file_path: Path to the audio file

    Returns:
        Evidence object
    """
    return Evidence(file_path)