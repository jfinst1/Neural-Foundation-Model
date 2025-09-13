"""
Neural Data Artifact Detection

Real-time artifact detection and removal for streaming neural data.
Handles eye blinks, muscle artifacts, electrode noise, and other common
artifacts in electrophysiological recordings.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import signal, stats
from sklearn.decomposition import FastICA
from typing import Dict, List, Optional, Tuple, Union
import warnings


class ArtifactDetector:
    """
    Multi-modal artifact detector for neural data.
    """
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,
        channels: Optional[List[int]] = None,
        real_time: bool = False,
        detection_methods: Optional[List[str]] = None
    ):
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.real_time = real_time
        
        # Default detection methods
        if detection_methods is None:
            detection_methods = ['amplitude', 'gradient', 'frequency', 'statistical']
        
        self.detection_methods = detection_methods
        
        # Initialize detectors
        self.detectors = {}
        self._initialize_detectors()
        
        # State for real-time processing
        if real_time:
            self.history_buffer = {}
            self.buffer_size = int(sampling_rate * 2)  # 2 seconds of history
    
    def _initialize_detectors(self):
        """Initialize individual artifact detectors."""
        if 'amplitude' in self.detection_methods:
            self.detectors['amplitude'] = AmplitudeArtifactDetector(
                sampling_rate=self.sampling_rate
            )
        
        if 'gradient' in self.detection_methods:
            self.detectors['gradient'] = GradientArtifactDetector(
                sampling_rate=self.sampling_rate
            )
        
        if 'frequency' in self.detection_methods:
            self.detectors['frequency'] = FrequencyArtifactDetector(
                sampling_rate=self.sampling_rate
            )
        
        if 'statistical' in self.detection_methods:
            self.detectors['statistical'] = StatisticalArtifactDetector(
                sampling_rate=self.sampling_rate
            )
        
        if 'muscle' in self.detection_methods:
            self.detectors['muscle'] = MuscleArtifactDetector(
                sampling_rate=self.sampling_rate
            )
        
        if 'eye_blink' in self.detection_methods:
            self.detectors['eye_blink'] = EyeBlinkDetector(
                sampling_rate=self.sampling_rate
            )
    
    def detect_artifacts(
        self, 
        data: np.ndarray, 
        return_details: bool = False
    ) -> Union[bool, Dict[str, bool]]:
        """
        Detect artifacts in neural data.
        
        Args:
            data: Input data (channels, time) or (time,) for single channel
            return_details: If True, return detailed results from each detector
            
        Returns:
            Boolean indicating artifact presence, or dict with detailed results
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        results = {}
        
        # Run each detector
        for method_name, detector in self.detectors.items():
            try:
                is_artifact = detector.detect(data)
                results[method_name] = is_artifact
            except Exception as e:
                warnings.warn(f"Detector {method_name} failed: {e}")
                results[method_name] = False
        
        if return_details:
            return results
        else:
            # Return True if any detector found artifacts
            return any(results.values())
    
    def clean_artifacts(
        self, 
        data: np.ndarray, 
        method: str = 'interpolation'
    ) -> np.ndarray:
        """
        Clean detected artifacts from data.
        
        Args:
            data: Input data (channels, time)
            method: Cleaning method ('interpolation', 'ica', 'zeroing')
            
        Returns:
            Cleaned data
        """
        # Detect artifact segments
        artifact_mask = self._get_artifact_mask(data)
        
        if not np.any(artifact_mask):
            return data  # No artifacts detected
        
        if method == 'interpolation':
            return self._interpolate_artifacts(data, artifact_mask)
        elif method == 'ica':
            return self._ica_artifact_removal(data)
        elif method == 'zeroing':
            cleaned_data = data.copy()
            cleaned_data[:, artifact_mask] = 0
            return cleaned_data
        else:
            raise ValueError(f"Unknown cleaning method: {method}")
    
    def _get_artifact_mask(self, data: np.ndarray) -> np.ndarray:
        """Get boolean mask indicating artifact time points."""
        mask = np.zeros(data.shape[1], dtype=bool)
        
        for detector in self.detectors.values():
            if hasattr(detector, 'get_artifact_mask'):
                detector_mask = detector.get_artifact_mask(data)
                mask |= detector_mask
        
        return mask
    
    def _interpolate_artifacts(self, data: np.ndarray, artifact_mask: np.ndarray) -> np.ndarray:
        """Interpolate over artifact segments."""
        cleaned_data = data.copy()
        
        for ch_idx in range(data.shape[0]):
            if np.any(artifact_mask):
                # Find artifact segments
                artifact_segments = self._find_continuous_segments(artifact_mask)
                
                for start, end in artifact_segments:
                    # Interpolate between clean segments
                    if start > 0 and end < data.shape[1] - 1:
                        # Linear interpolation
                        start_val = data[ch_idx, start - 1]
                        end_val = data[ch_idx, end + 1]
                        interp_vals = np.linspace(start_val, end_val, end - start + 1)
                        cleaned_data[ch_idx, start:end+1] = interp_vals
                    elif start == 0:
                        # Fill beginning with first clean value
                        cleaned_data[ch_idx, :end+1] = data[ch_idx, end + 1]
                    else:
                        # Fill end with last clean value
                        cleaned_data[ch_idx, start:] = data[ch_idx, start - 1]
        
        return cleaned_data
    
    def _ica_artifact_removal(self, data: np.ndarray) -> np.ndarray:
        """Remove artifacts using Independent Component Analysis."""
        if data.shape[0] < 2:
            warnings.warn("ICA requires at least 2 channels")
            return data
        
        # Apply FastICA
        ica = FastICA(n_components=min(data.shape[0], 20), random_state=42)
        
        try:
            # Fit ICA on transposed data (samples x channels)
            components = ica.fit_transform(data.T)
            
            # Identify artifact components (simplified heuristic)
            artifact_components = self._identify_artifact_components(components)
            
            # Zero out artifact components
            cleaned_components = components.copy()
            cleaned_components[:, artifact_components] = 0
            
            # Transform back to original space
            cleaned_data = ica.inverse_transform(cleaned_components).T
            
            return cleaned_data
        
        except Exception as e:
            warnings.warn(f"ICA artifact removal failed: {e}")
            return data
    
    def _identify_artifact_components(self, components: np.ndarray) -> List[int]:
        """Identify ICA components that likely represent artifacts."""
        artifact_components = []
        
        for i in range(components.shape[1]):
            component = components[:, i]
            
            # High kurtosis suggests artifacts (spiky signals)
            kurtosis = stats.kurtosis(component)
            if kurtosis > 5:
                artifact_components.append(i)
                continue
            
            # High power in muscle artifact frequency range (>30 Hz)
            freqs, psd = signal.welch(component, fs=self.sampling_rate, nperseg=256)
            muscle_power = np.mean(psd[freqs > 30])
            total_power = np.mean(psd)
            
            if muscle_power / total_power > 0.3:
                artifact_components.append(i)
        
        return artifact_components
    
    def _find_continuous_segments(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous True segments in boolean mask."""
        segments = []
        in_segment = False
        start_idx = 0
        
        for i, value in enumerate(mask):
            if value and not in_segment:
                start_idx = i
                in_segment = True
            elif not value and in_segment:
                segments.append((start_idx, i - 1))
                in_segment = False
        
        # Handle case where segment extends to end
        if in_segment:
            segments.append((start_idx, len(mask) - 1))
        
        return segments


class AmplitudeArtifactDetector:
    """Detects artifacts based on amplitude thresholds."""
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,
        threshold_std: float = 5.0,
        min_duration: float = 0.01  # 10ms
    ):
        self.sampling_rate = sampling_rate
        self.threshold_std = threshold_std
        self.min_duration_samples = int(min_duration * sampling_rate)
    
    def detect(self, data: np.ndarray) -> bool:
        """Detect amplitude-based artifacts."""
        # Compute per-channel thresholds
        thresholds = np.std(data, axis=1) * self.threshold_std
        
        # Check for threshold violations
        for ch_idx in range(data.shape[0]):
            violations = np.abs(data[ch_idx]) > thresholds[ch_idx]
            
            # Check for sustained violations
            if self._has_sustained_violations(violations):
                return True
        
        return False
    
    def get_artifact_mask(self, data: np.ndarray) -> np.ndarray:
        """Get boolean mask of artifact time points."""
        mask = np.zeros(data.shape[1], dtype=bool)
        
        # Compute per-channel thresholds
        thresholds = np.std(data, axis=1) * self.threshold_std
        
        for ch_idx in range(data.shape[0]):
            violations = np.abs(data[ch_idx]) > thresholds[ch_idx]
            
            # Extend violations to meet minimum duration
            extended_violations = self._extend_violations(violations)
            mask |= extended_violations
        
        return mask
    
    def _has_sustained_violations(self, violations: np.ndarray) -> bool:
        """Check if violations meet minimum duration requirement."""
        if not np.any(violations):
            return False
        
        # Find continuous violation segments
        diff = np.diff(violations.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if violations[0]:
            starts = np.concatenate([[0], starts])
        if violations[-1]:
            ends = np.concatenate([ends, [len(violations)]])
        
        # Check if any segment meets minimum duration
        for start, end in zip(starts, ends):
            if end - start >= self.min_duration_samples:
                return True
        
        return False
    
    def _extend_violations(self, violations: np.ndarray) -> np.ndarray:
        """Extend short violations to meet minimum duration."""
        extended = violations.copy()
        
        # Find violation segments
        diff = np.diff(violations.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if violations[0]:
            starts = np.concatenate([[0], starts])
        if violations[-1]:
            ends = np.concatenate([ends, [len(violations)]])
        
        # Extend short segments
        for start, end in zip(starts, ends):
            duration = end - start
            if 0 < duration < self.min_duration_samples:
                extension = self.min_duration_samples - duration
                new_start = max(0, start - extension // 2)
                new_end = min(len(extended), end + extension // 2)
                extended[new_start:new_end] = True
        
        return extended


class GradientArtifactDetector:
    """Detects artifacts based on signal gradient (rapid changes)."""
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,
        threshold_std: float = 4.0
    ):
        self.sampling_rate = sampling_rate
        self.threshold_std = threshold_std
    
    def detect(self, data: np.ndarray) -> bool:
        """Detect gradient-based artifacts."""
        # Compute gradient for each channel
        gradients = np.diff(data, axis=1)
        
        # Check for excessive gradients
        for ch_idx in range(gradients.shape[0]):
            grad_std = np.std(gradients[ch_idx])
            threshold = grad_std * self.threshold_std
            
            if np.any(np.abs(gradients[ch_idx]) > threshold):
                return True
        
        return False
    
    def get_artifact_mask(self, data: np.ndarray) -> np.ndarray:
        """Get boolean mask of artifact time points."""
        mask = np.zeros(data.shape[1], dtype=bool)
        
        # Compute gradient
        gradients = np.diff(data, axis=1)
        
        for ch_idx in range(gradients.shape[0]):
            grad_std = np.std(gradients[ch_idx])
            threshold = grad_std * self.threshold_std
            
            # Find excessive gradient points
            violations = np.abs(gradients[ch_idx]) > threshold
            
            # Extend mask to include neighboring points
            extended_mask = np.zeros(data.shape[1], dtype=bool)
            extended_mask[1:] |= violations  # Gradient violations
            extended_mask[:-1] |= violations  # Include previous point
            
            mask |= extended_mask
        
        return mask


class FrequencyArtifactDetector:
    """Detects artifacts based on frequency content."""
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,
        artifact_freq_ranges: Optional[List[Tuple[float, float]]] = None,
        power_threshold: float = 3.0
    ):
        self.sampling_rate = sampling_rate
        self.power_threshold = power_threshold
        
        # Default artifact frequency ranges
        if artifact_freq_ranges is None:
            artifact_freq_ranges = [
                (50.0, 60.0),    # Power line noise
                (100.0, 120.0),  # Power line harmonics
                (150.0, 200.0),  # High frequency artifacts
            ]
        
        self.artifact_freq_ranges = artifact_freq_ranges
    
    def detect(self, data: np.ndarray) -> bool:
        """Detect frequency-based artifacts."""
        # Compute power spectral density
        for ch_idx in range(data.shape[0]):
            freqs, psd = signal.welch(
                data[ch_idx], 
                fs=self.sampling_rate, 
                nperseg=min(256, data.shape[1] // 4)
            )
            
            # Check each artifact frequency range
            for low_freq, high_freq in self.artifact_freq_ranges:
                freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                
                if not np.any(freq_mask):
                    continue
                
                # Compute power in artifact range vs baseline
                artifact_power = np.mean(psd[freq_mask])
                
                # Baseline: power in adjacent frequency ranges
                baseline_mask = (
                    ((freqs >= low_freq - 10) & (freqs < low_freq)) |
                    ((freqs > high_freq) & (freqs <= high_freq + 10))
                )
                
                if np.any(baseline_mask):
                    baseline_power = np.mean(psd[baseline_mask])
                    if artifact_power > baseline_power * self.power_threshold:
                        return True
        
        return False


class StatisticalArtifactDetector:
    """Detects artifacts based on statistical properties."""
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,
        kurtosis_threshold: float = 5.0,
        skewness_threshold: float = 2.0
    ):
        self.sampling_rate = sampling_rate
        self.kurtosis_threshold = kurtosis_threshold
        self.skewness_threshold = skewness_threshold
    
    def detect(self, data: np.ndarray) -> bool:
        """Detect statistical artifacts."""
        for ch_idx in range(data.shape[0]):
            channel_data = data[ch_idx]
            
            # Check kurtosis (measure of tail heaviness)
            kurt = stats.kurtosis(channel_data)
            if kurt > self.kurtosis_threshold:
                return True
            
            # Check skewness (measure of asymmetry)
            skew = abs(stats.skew(channel_data))
            if skew > self.skewness_threshold:
                return True
        
        return False


class MuscleArtifactDetector:
    """Specialized detector for muscle artifacts (EMG contamination)."""
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,
        muscle_freq_range: Tuple[float, float] = (30.0, 100.0),
        power_ratio_threshold: float = 2.0
    ):
        self.sampling_rate = sampling_rate
        self.muscle_freq_range = muscle_freq_range
        self.power_ratio_threshold = power_ratio_threshold
    
    def detect(self, data: np.ndarray) -> bool:
        """Detect muscle artifacts based on high-frequency power."""
        for ch_idx in range(data.shape[0]):
            freqs, psd = signal.welch(
                data[ch_idx],
                fs=self.sampling_rate,
                nperseg=min(256, data.shape[1] // 4)
            )
            
            # Power in muscle frequency range
            muscle_mask = (freqs >= self.muscle_freq_range[0]) & (freqs <= self.muscle_freq_range[1])
            muscle_power = np.mean(psd[muscle_mask]) if np.any(muscle_mask) else 0
            
            # Power in neural frequency range (1-30 Hz)
            neural_mask = (freqs >= 1.0) & (freqs <= 30.0)
            neural_power = np.mean(psd[neural_mask]) if np.any(neural_mask) else 1e-10
            
            # Check power ratio
            if muscle_power / neural_power > self.power_ratio_threshold:
                return True
        
        return False


class EyeBlinkDetector:
    """Specialized detector for eye blink artifacts."""
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,
        blink_duration_range: Tuple[float, float] = (0.1, 0.5),  # 100-500ms
        amplitude_threshold: float = 3.0,
        frontal_channels: Optional[List[int]] = None
    ):
        self.sampling_rate = sampling_rate
        self.blink_duration_range = blink_duration_range
        self.amplitude_threshold = amplitude_threshold
        self.frontal_channels = frontal_channels or [0, 1]  # Default to first 2 channels
    
    def detect(self, data: np.ndarray) -> bool:
        """Detect eye blink artifacts."""
        # Focus on frontal channels (where eye blinks are most prominent)
        frontal_channels = [ch for ch in self.frontal_channels if ch < data.shape[0]]
        
        if not frontal_channels:
            return False
        
        frontal_data = data[frontal_channels]
        
        # Look for characteristic eye blink pattern:
        # 1. Negative deflection in frontal channels
        # 2. Specific duration
        # 3. Symmetric pattern
        
        for ch_idx in range(frontal_data.shape[0]):
            channel_data = frontal_data[ch_idx]
            
            # Detect negative peaks
            peaks, properties = signal.find_peaks(
                -channel_data,  # Invert for negative peaks
                height=np.std(channel_data) * self.amplitude_threshold,
                width=self.blink_duration_range[0] * self.sampling_rate
            )
            
            # Check if any peaks have eye blink characteristics
            for peak_idx in peaks:
                peak_width = properties['widths'][list(peaks).index(peak_idx)]
                peak_duration = peak_width / self.sampling_rate
                
                if (self.blink_duration_range[0] <= peak_duration <= 
                    self.blink_duration_range[1]):
                    return True
        
        return False


class RealTimeArtifactProcessor:
    """
    Real-time artifact processing for streaming neural data.
    """
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,
        buffer_size: float = 2.0,  # seconds
        num_channels: int = 64,
        detection_methods: Optional[List[str]] = None
    ):
        self.sampling_rate = sampling_rate
        self.buffer_size = int(buffer_size * sampling_rate)
        self.num_channels = num_channels
        
        # Circular buffer for incoming data
        self.data_buffer = np.zeros((num_channels, self.buffer_size))
        self.buffer_ptr = 0
        self.buffer_full = False
        
        # Artifact detector
        self.detector = ArtifactDetector(
            sampling_rate=sampling_rate,
            channels=list(range(num_channels)),
            real_time=True,
            detection_methods=detection_methods
        )
        
        # Artifact flags and history
        self.artifact_history = []
        self.current_artifact_state = False
    
    def process_sample(self, sample: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Process single sample and detect artifacts.
        
        Args:
            sample: Single sample (num_channels,)
            
        Returns:
            Tuple of (processed_sample, is_artifact)
        """
        # Add sample to buffer
        self.data_buffer[:, self.buffer_ptr] = sample
        self.buffer_ptr = (self.buffer_ptr + 1) % self.buffer_size
        
        if not self.buffer_full and self.buffer_ptr == 0:
            self.buffer_full = True
        
        # Only process if buffer has enough data
        if not self.buffer_full:
            return sample, False
        
        # Get recent data for artifact detection
        recent_data = self._get_recent_data(int(0.5 * self.sampling_rate))  # 500ms window
        
        # Detect artifacts
        is_artifact = self.detector.detect_artifacts(recent_data)
        
        # Update artifact state
        self.artifact_history.append(is_artifact)
        if len(self.artifact_history) > 10:  # Keep last 10 decisions
            self.artifact_history.pop(0)
        
        # Smooth artifact decisions (avoid flickering)
        artifact_ratio = sum(self.artifact_history) / len(self.artifact_history)
        self.current_artifact_state = artifact_ratio > 0.3
        
        # Apply simple cleaning if artifact detected
        if self.current_artifact_state:
            # Use median of recent clean samples
            clean_samples = self._get_clean_samples()
            if clean_samples is not None:
                processed_sample = np.median(clean_samples, axis=1)
            else:
                processed_sample = np.zeros_like(sample)  # Zero out if no clean samples
        else:
            processed_sample = sample
        
        return processed_sample, self.current_artifact_state
    
    def _get_recent_data(self, num_samples: int) -> np.ndarray:
        """Get recent data from circular buffer."""
        if not self.buffer_full and num_samples > self.buffer_ptr:
            num_samples = self.buffer_ptr
        
        if num_samples > self.buffer_size:
            num_samples = self.buffer_size
        
        if self.buffer_full:
            start_idx = (self.buffer_ptr - num_samples) % self.buffer_size
            if start_idx + num_samples <= self.buffer_size:
                return self.data_buffer[:, start_idx:start_idx + num_samples]
            else:
                # Wrap around
                part1 = self.data_buffer[:, start_idx:]
                part2 = self.data_buffer[:, :start_idx + num_samples - self.buffer_size]
                return np.concatenate([part1, part2], axis=1)
        else:
            start_idx = max(0, self.buffer_ptr - num_samples)
            return self.data_buffer[:, start_idx:self.buffer_ptr]
    
    def _get_clean_samples(self, num_samples: int = 10) -> Optional[np.ndarray]:
        """Get recent clean samples for interpolation."""
        # Look back through buffer for clean samples
        clean_samples = []
        
        for i in range(min(len(self.artifact_history), self.buffer_size)):
            history_idx = len(self.artifact_history) - 1 - i
            if history_idx >= 0 and not self.artifact_history[history_idx]:
                buffer_idx = (self.buffer_ptr - 1 - i) % self.buffer_size
                clean_samples.append(self.data_buffer[:, buffer_idx])
                
                if len(clean_samples) >= num_samples:
                    break
        
        if clean_samples:
            return np.column_stack(clean_samples)
        else:
            return None