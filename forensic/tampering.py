"""
Tampering detection module for Audio Forensic Analyzer.
Detects audio discontinuities and potential splicing.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class TamperingDetector:
    """Detects audio tampering through feature discontinuity analysis."""

    def __init__(
        self,
        window_size: float = 1.0,
        overlap: float = 0.5,
        anomaly_threshold: float = 2.0,
        use_isolation_forest: bool = False,
        contamination: float = 0.1
    ):
        """
        Initialize tampering detector.

        Args:
            window_size: Size of analysis window in seconds
            overlap: Overlap between windows (0.0 to 1.0)
            anomaly_threshold: Threshold for anomaly detection (standard deviations)
            use_isolation_forest: Whether to use Isolation Forest for detection
            contamination: Expected proportion of outliers for Isolation Forest
        """
        self.window_size = window_size
        self.overlap = overlap
        self.anomaly_threshold = anomaly_threshold
        self.use_isolation_forest = use_isolation_forest
        self.contamination = contamination
        self.isolation_forest = None

        if self.use_isolation_forest:
            self.isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )

    def _create_windows(
        self,
        features: Dict[str, np.ndarray],
        hop_length: int
    ) -> Tuple[List[Dict[str, float]], List[float]]:
        """
        Create overlapping windows of features.

        Args:
            features: Dictionary of feature arrays
            hop_length: Hop length used for feature extraction

        Returns:
            Tuple of (window_features_list, window_times_list)
        """
        # Get the minimum length among all features
        min_length = min(len(v) for v in features.values())

        # Calculate window and hop samples
        window_samples = int(self.window_size * 100)  # Approximate conversion
        hop_samples = int(window_samples * (1 - self.overlap))

        # Ensure we have reasonable values
        if window_samples < 1:
            window_samples = 1
        if hop_samples < 1:
            hop_samples = 1

        windows = []
        window_times = []

        # Create windows
        for i in range(0, min_length - window_samples + 1, hop_samples):
            window_features = {}
            for feature_name, feature_array in features.items():
                # Extract window and compute mean
                window_data = feature_array[i:i + window_samples]
                window_features[feature_name] = float(np.mean(window_data))

            windows.append(window_features)
            # Approximate time in seconds (this is rough but works for display)
            window_times.append(i * hop_length / 100)  # Rough conversion

        return windows, window_times

    def _extract_feature_vectors(self, windows: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract feature vectors from windows for analysis.

        Args:
            windows: List of window feature dictionaries

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if not windows:
            return np.array([]), []

        # Get feature names from first window
        feature_names = list(windows[0].keys())

        # Create feature matrix
        feature_matrix = np.array([[window[fname] for fname in feature_names] for window in windows])

        return feature_matrix, feature_names

    def _calculate_window_differences(
        self,
        feature_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate differences between adjacent windows.

        Args:
            feature_matrix: Matrix of features (windows x features)

        Returns:
            Matrix of differences between adjacent windows
        """
        if len(feature_matrix) < 2:
            return np.array([])

        # Calculate differences between consecutive windows
        diffs = np.diff(feature_matrix, axis=0)
        return diffs

    def _detect_anomalies_threshold(
        self,
        diffs: np.ndarray,
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies using threshold-based approach.

        Args:
            diffs: Difference matrix between adjacent windows
            feature_names: List of feature names

        Returns:
            List of anomaly dictionaries
        """
        anomalies = []

        if len(diffs) == 0:
            return anomalies

        # Calculate anomaly score for each difference vector
        for i, diff_vector in enumerate(diffs):
            # Calculate normalized anomaly score (Euclidean norm)
            anomaly_score = np.linalg.norm(diff_vector)

            # Calculate threshold based on median absolute deviation (robust)
            if len(diffs) > 1:
                # Use robust statistics for threshold
                median_abs_dev = np.median(np.abs(diffs - np.median(diffs, axis=0)), axis=0)
                threshold = np.sqrt(np.sum((self.anomaly_threshold * median_abs_dev) ** 2))
            else:
                # Fallback for small datasets
                threshold = self.anomaly_threshold * np.sqrt(len(feature_names))

            is_anomaly = anomaly_score > threshold

            if is_anomaly:
                # Find which features contributed most to the anomaly
                feature_contributions = np.abs(diff_vector)
                max_contrib_idx = np.argmax(feature_contributions)
                max_feature = feature_names[max_contrib_idx]
                max_contribution = feature_contributions[max_contrib_idx]

                anomalies.append({
                    'window_index': i + 1,  # +1 because diffs are between windows i and i+1
                    'anomaly_score': float(anomaly_score),
                    'threshold': float(threshold),
                    'is_anomaly': True,
                    'max_contributing_feature': max_feature,
                    'max_contribution': float(max_contribution),
                    'feature_differences': {
                        fname: float(diff) for fname, diff in zip(feature_names, diff_vector)
                    }
                })

        return anomalies

    def _detect_anomalies_isolation_forest(
        self,
        feature_matrix: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies using Isolation Forest.

        Args:
            feature_matrix: Matrix of features (windows x features)

        Returns:
            List of anomaly dictionaries
        """
        anomalies = []

        if len(feature_matrix) < 2:
            return anomalies

        # Fit Isolation Forest
        if self.isolation_forest is not None:
            predictions = self.isolation_forest.fit_predict(feature_matrix)
            anomaly_scores = self.isolation_forest.decision_function(feature_matrix)

            # Convert scores to anomaly format (lower scores = more anomalous)
            for i, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
                is_anomaly = pred == -1  # Isolation Forest returns -1 for anomalies

                if is_anomaly:
                    anomalies.append({
                        'window_index': i,
                        'anomaly_score': float(-score),  # Invert score so higher = more anomalous
                        'threshold': 0.0,  # Not applicable for Isolation Forest
                        'is_anomaly': True,
                        'detection_method': 'isolation_forest'
                    })

        return anomalies

    def detect_tampering(
        self,
        features: Dict[str, np.ndarray],
        hop_length: int
    ) -> Dict[str, Any]:
        """
        Detect tampering in audio features.

        Args:
            features: Dictionary of feature arrays (from FeatureExtractor)
            hop_length: Hop length used for feature extraction

        Returns:
            Dictionary containing detection results
        """
        # Create windows
        windows, window_times = self._create_windows(features, hop_length)

        if len(windows) < 2:
            return {
                'anomalies': [],
                'window_times': [],
                'num_windows': len(windows),
                'message': 'Insufficient data for tampering detection'
            }

        # Extract feature vectors
        feature_matrix, feature_names = self._extract_feature_vectors(windows)

        # Detect anomalies based on method
        if self.use_isolation_forest:
            anomalies = self._detect_anomalies_isolation_forest(feature_matrix)
        else:
            # Calculate window differences and detect anomalies
            diffs = self._calculate_window_differences(feature_matrix)
            anomalies = self._detect_anomalies_threshold(diffs, feature_names)

        # Convert window indices to approximate times
        suspicious_timestamps = []
        for anomaly in anomalies:
            if 'window_index' in anomaly:
                # Approximate time calculation
                idx = anomaly['window_index']
                if idx < len(window_times):
                    suspicious_timestamps.append(window_times[idx])
                else:
                    # Fallback approximation
                    suspicious_timestamps.append(idx * self.window_size)

        return {
            'anomalies': anomalies,
            'window_times': window_times,
            'num_windows': len(windows),
            'suspicious_timestamps': suspicious_timestamps,
            'max_anomaly_score': max([a.get('anomaly_score', 0) for a in anomalies]) if anomalies else 0
        }


def create_tampering_detector(
    window_size: float = 1.0,
    overlap: float = 0.5,
    anomaly_threshold: float = 2.0,
    use_isolation_forest: bool = False,
    contamination: float = 0.1
) -> TamperingDetector:
    """
    Factory function to create TamperingDetector object.

    Args:
        window_size: Size of analysis window in seconds
        overlap: Overlap between windows (0.0 to 1.0)
        anomaly_threshold: Threshold for anomaly detection (standard deviations)
        use_isolation_forest: Whether to use Isolation Forest for detection
        contamination: Expected proportion of outliers for Isolation Forest

    Returns:
        TamperingDetector object
    """
    return TamperingDetector(
        window_size=window_size,
        overlap=overlap,
        anomaly_threshold=anomaly_threshold,
        use_isolation_forest=use_isolation_forest,
        contamination=contamination
    )