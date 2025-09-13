#!/usr/bin/env python3
"""
Neural Data Preprocessing Pipeline

Comprehensive preprocessing for electrophysiological neural data including
filtering, artifact removal, segmentation, and quality validation.
Optimized for large-scale neural foundation model training.
"""

import argparse
import logging
import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import sys

import h5py
import mne
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_foundation.signal_processing.filters import CausalButterworthFilter, NotchFilter, MultiChannelCAR
from neural_foundation.signal_processing.artifacts import ArtifactDetector
from neural_foundation.privacy.anonymization import NeuralDataAnonymizer


class NeuralDataPreprocessor:
    """
    Comprehensive neural data preprocessing pipeline.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config_path: Path to preprocessing configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize processing components
        self._init_filters()
        self._init_artifact_detector()
        self._init_anonymizer()
        
        # Statistics tracking
        self.processing_stats = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_duration_hours': 0.0,
            'artifacts_detected': 0,
            'channels_rejected': 0
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load preprocessing configuration."""
        default_config = {
            'sampling_rate': 1000.0,
            'target_sampling_rate': 1000.0,
            'channels': {
                'select_channels': None,  # None = use all
                'exclude_channels': [],
                'reference': 'average',  # 'average', 'mastoid', None
            },
            'filtering': {
                'enable_bandpass': True,
                'low_freq': 0.5,
                'high_freq': 100.0,
                'filter_order': 4,
                'enable_notch': True,
                'notch_freq': 50.0,
                'notch_quality': 30.0,
            },
            'artifacts': {
                'enable_detection': True,
                'methods': ['amplitude', 'gradient', 'frequency'],
                'amplitude_threshold': 5.0,
                'reject_channels': True,
                'interpolate_bad_channels': True,
            },
            'segmentation': {
                'segment_length': 10.0,  # seconds
                'overlap': 0.0,  # seconds
                'min_segment_length': 5.0,  # seconds
            },
            'normalization': {
                'method': 'robust',  # 'zscore', 'robust', 'minmax', None
                'per_channel': True,
                'clip_outliers': True,
                'outlier_std': 5.0,
            },
            'quality_control': {
                'min_duration': 60.0,  # seconds
                'max_artifact_ratio': 0.5,
                'min_good_channels': 32,
            },
            'privacy': {
                'enable_anonymization': False,
                'method': 'differential_privacy',
                'noise_level': 0.1,
            },
            'output': {
                'format': 'hdf5',  # 'hdf5', 'numpy', 'mat'
                'compression': 'gzip',
                'chunk_size': 1000,  # samples
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # Merge configs (user overrides default)
            default_config.update(user_config)
        
        return default_config
    
    def _init_filters(self):
        """Initialize signal processing filters."""
        if self.config['filtering']['enable_bandpass']:
            self.bandpass_filter = CausalButterworthFilter(
                low_freq=self.config['filtering']['low_freq'],
                high_freq=self.config['filtering']['high_freq'],
                sampling_rate=self.config['sampling_rate'],
                order=self.config['filtering']['filter_order']
            )
        else:
            self.bandpass_filter = None
        
        if self.config['filtering']['enable_notch']:
            self.notch_filter = NotchFilter(
                notch_freq=self.config['filtering']['notch_freq'],
                quality_factor=self.config['filtering']['notch_quality'],
                sampling_rate=self.config['sampling_rate']
            )
        else:
            self.notch_filter = None
        
        # Common average reference
        if self.config['channels']['reference'] == 'average':
            self.car_filter = MultiChannelCAR()
        else:
            self.car_filter = None
    
    def _init_artifact_detector(self):
        """Initialize artifact detection."""
        if self.config['artifacts']['enable_detection']:
            self.artifact_detector = ArtifactDetector(
                sampling_rate=self.config['sampling_rate'],
                detection_methods=self.config['artifacts']['methods']
            )
        else:
            self.artifact_detector = None
    
    def _init_anonymizer(self):
        """Initialize data anonymization."""
        if self.config['privacy']['enable_anonymization']:
            self.anonymizer = NeuralDataAnonymizer(
                method=self.config['privacy']['method'],
                noise_level=self.config['privacy']['noise_level']
            )
        else:
            self.anonymizer = None
    
    def preprocess_file(self, input_path: Path, output_dir: Path) -> Dict[str, Union[bool, str, float]]:
        """
        Preprocess a single neural data file.
        
        Args:
            input_path: Path to input file
            output_dir: Output directory
            
        Returns:
            Dictionary with processing results and statistics
        """
        result = {
            'success': False,
            'input_file': str(input_path),
            'output_file': None,
            'duration_seconds': 0.0,
            'num_channels': 0,
            'num_segments': 0,
            'artifacts_ratio': 0.0,
            'rejected_channels': 0,
            'error_message': None
        }
        
        try:
            # Load neural data
            self.logger.info(f"Processing {input_path.name}...")
            raw_data, metadata = self._load_neural_data(input_path)
            
            if raw_data is None:
                result['error_message'] = "Failed to load data"
                return result
            
            result['duration_seconds'] = raw_data.shape[1] / metadata['sampling_rate']
            result['num_channels'] = raw_data.shape[0]
            
            # Quality control checks
            if not self._passes_quality_control(raw_data, metadata):
                result['error_message'] = "Failed quality control"
                return result
            
            # Channel selection and rejection
            good_channels, rejected_channels = self._select_channels(raw_data, metadata)
            raw_data = raw_data[good_channels]
            result['rejected_channels'] = len(rejected_channels)
            result['num_channels'] = raw_data.shape[0]
            
            # Signal preprocessing pipeline
            processed_data = self._apply_signal_processing(raw_data, metadata)
            
            # Artifact detection and cleaning
            if self.artifact_detector:
                artifact_mask = self._detect_artifacts(processed_data)
                processed_data = self._clean_artifacts(processed_data, artifact_mask)
                result['artifacts_ratio'] = np.mean(artifact_mask) if len(artifact_mask) > 0 else 0.0
            
            # Normalization
            processed_data = self._normalize_data(processed_data)
            
            # Privacy protection
            if self.anonymizer:
                processed_data = self.anonymizer.anonymize(processed_data)
            
            # Segmentation
            segments, segment_metadata = self._segment_data(processed_data, metadata)
            result['num_segments'] = len(segments)
            
            # Save processed data
            output_path = self._save_processed_data(
                segments, segment_metadata, input_path, output_dir
            )
            result['output_file'] = str(output_path)
            result['success'] = True
            
            self.logger.info(f"Successfully processed {input_path.name}: "
                           f"{result['num_segments']} segments, "
                           f"{result['artifacts_ratio']:.1%} artifacts")
            
        except Exception as e:
            self.logger.error(f"Error processing {input_path.name}: {str(e)}")
            result['error_message'] = str(e)
        
        return result
    
    def _load_neural_data(self, input_path: Path) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Load neural data from various formats.
        
        Args:
            input_path: Path to input file
            
        Returns:
            Tuple of (data_array, metadata)
        """
        metadata = {'sampling_rate': self.config['sampling_rate']}
        
        try:
            if input_path.suffix.lower() in ['.fif', '.fiff']:
                # MNE-Python format
                raw = mne.io.read_raw_fif(str(input_path), preload=True, verbose=False)
                data = raw.get_data()
                metadata.update({
                    'sampling_rate': raw.info['sfreq'],
                    'channel_names': raw.ch_names,
                    'subject_id': raw.info.get('subject_info', {}).get('id', 'unknown')
                })
                
            elif input_path.suffix.lower() == '.h5':
                # HDF5 format
                with h5py.File(input_path, 'r') as f:
                    data = f['data'][:]
                    if 'metadata' in f.attrs:
                        stored_metadata = dict(f.attrs)
                        metadata.update(stored_metadata)
                        
            elif input_path.suffix.lower() in ['.edf', '.bdf']:
                # EDF/BDF format
                raw = mne.io.read_raw_edf(str(input_path), preload=True, verbose=False)
                data = raw.get_data()
                metadata.update({
                    'sampling_rate': raw.info['sfreq'],
                    'channel_names': raw.ch_names
                })
                
            elif input_path.suffix.lower() == '.mat':
                # MATLAB format
                from scipy.io import loadmat
                mat_data = loadmat(str(input_path))
                # Look for common variable names
                for key in ['data', 'eeg', 'signals']:
                    if key in mat_data:
                        data = mat_data[key]
                        break
                else:
                    # Use the largest numeric array
                    arrays = [(k, v) for k, v in mat_data.items() 
                             if isinstance(v, np.ndarray) and v.ndim >= 2]
                    if arrays:
                        data = max(arrays, key=lambda x: x[1].size)[1]
                    else:
                        raise ValueError("No suitable data array found in .mat file")
                        
            else:
                self.logger.warning(f"Unsupported file format: {input_path.suffix}")
                return None, metadata
            
            # Ensure data is in correct format (channels, time)
            if data.ndim != 2:
                raise ValueError(f"Expected 2D data array, got {data.ndim}D")
            
            # Convert to float32 for memory efficiency
            data = data.astype(np.float32)
            
            return data, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load {input_path}: {str(e)}")
            return None, metadata
    
    def _passes_quality_control(self, data: np.ndarray, metadata: Dict) -> bool:
        """Check if data passes quality control criteria."""
        duration = data.shape[1] / metadata['sampling_rate']
        
        # Check minimum duration
        if duration < self.config['quality_control']['min_duration']:
            self.logger.warning(f"Data too short: {duration:.1f}s < "
                              f"{self.config['quality_control']['min_duration']}s")
            return False
        
        # Check minimum number of channels
        if data.shape[0] < self.config['quality_control']['min_good_channels']:
            self.logger.warning(f"Too few channels: {data.shape[0]} < "
                              f"{self.config['quality_control']['min_good_channels']}")
            return False
        
        # Check for excessive artifacts
        if self.artifact_detector:
            artifacts_detected = self.artifact_detector.detect_artifacts(data)
            if artifacts_detected:
                # Estimate artifact ratio (simplified)
                artifact_mask = self._detect_artifacts(data)
                artifact_ratio = np.mean(artifact_mask) if len(artifact_mask) > 0 else 0.0
                
                if artifact_ratio > self.config['quality_control']['max_artifact_ratio']:
                    self.logger.warning(f"Too many artifacts: {artifact_ratio:.1%} > "
                                      f"{self.config['quality_control']['max_artifact_ratio']:.1%}")
                    return False
        
        return True
    
    def _select_channels(self, data: np.ndarray, metadata: Dict) -> Tuple[List[int], List[int]]:
        """Select good channels and identify bad ones."""
        num_channels = data.shape[0]
        all_channels = list(range(num_channels))
        
        # Start with specified channels or all channels
        if self.config['channels']['select_channels']:
            selected = self.config['channels']['select_channels']
            good_channels = [ch for ch in selected if ch < num_channels]
        else:
            good_channels = all_channels.copy()
        
        # Remove excluded channels
        excluded = self.config['channels']['exclude_channels']
        good_channels = [ch for ch in good_channels if ch not in excluded]
        
        # Automatic bad channel detection
        if self.config['artifacts']['reject_channels']:
            bad_channels = self._detect_bad_channels(data[good_channels])
            good_channels = [ch for i, ch in enumerate(good_channels) if i not in bad_channels]
        
        rejected_channels = [ch for ch in all_channels if ch not in good_channels]
        
        return good_channels, rejected_channels
    
    def _detect_bad_channels(self, data: np.ndarray) -> List[int]:
        """Detect bad channels based on statistical criteria."""
        bad_channels = []
        
        for ch_idx in range(data.shape[0]):
            channel_data = data[ch_idx]
            
            # Check for flat channels (minimal variance)
            if np.var(channel_data) < 1e-6:
                bad_channels.append(ch_idx)
                continue
            
            # Check for excessive noise (high variance)
            channel_std = np.std(channel_data)
            median_std = np.median([np.std(data[i]) for i in range(data.shape[0])])
            if channel_std > 5 * median_std:
                bad_channels.append(ch_idx)
                continue
            
            # Check for excessive artifacts
            if self.artifact_detector:
                if self.artifact_detector.detect_artifacts(channel_data.reshape(1, -1)):
                    bad_channels.append(ch_idx)
        
        if bad_channels:
            self.logger.info(f"Detected {len(bad_channels)} bad channels: {bad_channels}")
        
        return bad_channels
    
    def _apply_signal_processing(self, data: np.ndarray, metadata: Dict) -> np.ndarray:
        """Apply signal processing pipeline."""
        processed = data.copy()
        
        # Resampling if needed
        current_sr = metadata['sampling_rate']
        target_sr = self.config['target_sampling_rate']
        
        if abs(current_sr - target_sr) > 1:  # Allow 1 Hz tolerance
            self.logger.info(f"Resampling from {current_sr} Hz to {target_sr} Hz")
            processed = signal.resample(
                processed, 
                int(processed.shape[1] * target_sr / current_sr),
                axis=1
            )
            metadata['sampling_rate'] = target_sr
        
        # Apply bandpass filter
        if self.bandpass_filter:
            self.logger.debug("Applying bandpass filter")
            processed = self.bandpass_filter.apply(processed)
        
        # Apply notch filter
        if self.notch_filter:
            self.logger.debug("Applying notch filter")
            processed = self.notch_filter.apply(processed)
        
        # Apply common average reference
        if self.car_filter:
            self.logger.debug("Applying common average reference")
            processed = self.car_filter.apply(processed)
        
        return processed
    
    def _detect_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Detect artifacts in the data."""
        if not self.artifact_detector:
            return np.zeros(data.shape[1], dtype=bool)
        
        # Get artifact mask for each channel and combine
        artifact_mask = np.zeros(data.shape[1], dtype=bool)
        
        for ch_idx in range(data.shape[0]):
            channel_mask = self.artifact_detector.get_artifact_mask(
                data[ch_idx:ch_idx+1]  # Single channel as 2D array
            )
            artifact_mask |= channel_mask
        
        return artifact_mask
    
    def _clean_artifacts(self, data: np.ndarray, artifact_mask: np.ndarray) -> np.ndarray:
        """Clean detected artifacts from data."""
        if not np.any(artifact_mask):
            return data
        
        cleaned = data.copy()
        
        # Simple interpolation over artifact segments
        for ch_idx in range(data.shape[0]):
            channel_data = cleaned[ch_idx]
            
            # Find artifact segments
            artifact_segments = self._find_continuous_segments(artifact_mask)
            
            for start, end in artifact_segments:
                if start > 0 and end < len(channel_data) - 1:
                    # Linear interpolation
                    start_val = channel_data[start - 1]
                    end_val = channel_data[end + 1]
                    interp_vals = np.linspace(start_val, end_val, end - start + 1)
                    cleaned[ch_idx, start:end+1] = interp_vals
                elif start == 0:
                    # Fill beginning with first clean value
                    cleaned[ch_idx, :end+1] = channel_data[end + 1]
                else:
                    # Fill end with last clean value
                    cleaned[ch_idx, start:] = channel_data[start - 1]
        
        return cleaned
    
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
        
        if in_segment:
            segments.append((start_idx, len(mask) - 1))
        
        return segments
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data according to configuration."""
        if self.config['normalization']['method'] is None:
            return data
        
        normalized = data.copy()
        
        if self.config['normalization']['per_channel']:
            # Normalize each channel separately
            for ch_idx in range(data.shape[0]):
                channel_data = normalized[ch_idx]
                normalized[ch_idx] = self._normalize_channel(channel_data)
        else:
            # Global normalization
            normalized = self._normalize_channel(normalized.flatten()).reshape(data.shape)
        
        # Clip outliers if enabled
        if self.config['normalization']['clip_outliers']:
            outlier_threshold = self.config['normalization']['outlier_std']
            normalized = np.clip(normalized, -outlier_threshold, outlier_threshold)
        
        return normalized
    
    def _normalize_channel(self, data: np.ndarray) -> np.ndarray:
        """Normalize a single channel of data."""
        method = self.config['normalization']['method']
        
        if method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            return (data - mean) / (std + 1e-8)
        
        elif method == 'robust':
            scaler = RobustScaler()
            return scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        elif method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            return (data - min_val) / (max_val - min_val + 1e-8)
        
        else:
            return data
    
    def _segment_data(self, data: np.ndarray, metadata: Dict) -> Tuple[List[np.ndarray], List[Dict]]:
        """Segment continuous data into fixed-length segments."""
        sampling_rate = metadata['sampling_rate']
        segment_length = int(self.config['segmentation']['segment_length'] * sampling_rate)
        overlap_length = int(self.config['segmentation']['overlap'] * sampling_rate)
        min_length = int(self.config['segmentation']['min_segment_length'] * sampling_rate)
        
        step_size = segment_length - overlap_length
        segments = []
        segment_metadata = []
        
        start_idx = 0
        segment_idx = 0
        
        while start_idx + min_length <= data.shape[1]:
            end_idx = min(start_idx + segment_length, data.shape[1])
            
            # Extract segment
            segment = data[:, start_idx:end_idx]
            
            # Skip if too short
            if segment.shape[1] < min_length:
                break
            
            segments.append(segment)
            
            # Create segment metadata
            seg_metadata = metadata.copy()
            seg_metadata.update({
                'segment_idx': segment_idx,
                'start_time': start_idx / sampling_rate,
                'end_time': end_idx / sampling_rate,
                'duration': segment.shape[1] / sampling_rate,
            })
            segment_metadata.append(seg_metadata)
            
            start_idx += step_size
            segment_idx += 1
        
        return segments, segment_metadata
    
    def _save_processed_data(
        self,
        segments: List[np.ndarray],
        segment_metadata: List[Dict],
        input_path: Path,
        output_dir: Path
    ) -> Path:
        """Save processed data segments."""
        output_format = self.config['output']['format']
        base_name = input_path.stem
        
        if output_format == 'hdf5':
            output_path = output_dir / f"{base_name}_processed.h5"
            self._save_hdf5(segments, segment_metadata, output_path)
        
        elif output_format == 'numpy':
            output_path = output_dir / f"{base_name}_processed.npz"
            self._save_numpy(segments, segment_metadata, output_path)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return output_path
    
    def _save_hdf5(self, segments: List[np.ndarray], metadata: List[Dict], output_path: Path):
        """Save data in HDF5 format."""
        with h5py.File(output_path, 'w') as f:
            # Save segments
            for i, segment in enumerate(segments):
                f.create_dataset(
                    f'segment_{i:04d}',
                    data=segment,
                    compression=self.config['output']['compression'],
                    chunks=True
                )
            
            # Save metadata
            for i, meta in enumerate(metadata):
                grp = f.create_group(f'metadata_{i:04d}')
                for key, value in meta.items():
                    if isinstance(value, (str, int, float, bool)):
                        grp.attrs[key] = value
                    elif isinstance(value, (list, tuple)):
                        grp.attrs[key] = np.array(value)
            
            # Global metadata
            f.attrs['num_segments'] = len(segments)
            f.attrs['preprocessing_config'] = str(self.config)
    
    def _save_numpy(self, segments: List[np.ndarray], metadata: List[Dict], output_path: Path):
        """Save data in NumPy format."""
        save_dict = {}
        
        # Save segments
        for i, segment in enumerate(segments):
            save_dict[f'segment_{i:04d}'] = segment
        
        # Save metadata (simplified)
        save_dict['metadata'] = metadata
        save_dict['num_segments'] = len(segments)
        
        np.savez_compressed(output_path, **save_dict)
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        num_workers: int = None
    ) -> Dict[str, int]:
        """
        Process all files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path  
            num_workers: Number of parallel workers
            
        Returns:
            Dictionary with processing statistics
        """
        # Find all neural data files
        supported_extensions = ['.fif', '.fiff', '.h5', '.edf', '.bdf', '.mat']
        input_files = []
        
        for ext in supported_extensions:
            input_files.extend(input_dir.glob(f'*{ext}'))
            input_files.extend(input_dir.glob(f'**/*{ext}'))
        
        if not input_files:
            self.logger.warning(f"No supported files found in {input_dir}")
            return self.processing_stats
        
        self.logger.info(f"Found {len(input_files)} files to process")
        self.processing_stats['total_files'] = len(input_files)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files
        if num_workers is None:
            num_workers = min(mp.cpu_count() - 1, 8)
        
        if num_workers > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(self.preprocess_file, input_file, output_dir): input_file
                    for input_file in input_files
                }
                
                with tqdm(total=len(futures), desc="Processing files") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        self._update_stats(result)
                        pbar.update(1)
                        
                        if result['success']:
                            pbar.set_postfix(success=self.processing_stats['successful_files'])
                        else:
                            pbar.set_postfix(
                                success=self.processing_stats['successful_files'],
                                failed=self.processing_stats['failed_files']
                            )
        else:
            # Sequential processing
            for input_file in tqdm(input_files, desc="Processing files"):
                result = self.preprocess_file(input_file, output_dir)
                self._update_stats(result)
        
        # Save processing report
        self._save_processing_report(output_dir)
        
        return self.processing_stats
    
    def _update_stats(self, result: Dict):
        """Update processing statistics."""
        if result['success']:
            self.processing_stats['successful_files'] += 1
            self.processing_stats['total_duration_hours'] += result['duration_seconds'] / 3600
        else:
            self.processing_stats['failed_files'] += 1
        
        self.processing_stats['artifacts_detected'] += result.get('artifacts_ratio', 0)
        self.processing_stats['channels_rejected'] += result.get('rejected_channels', 0)
    
    def _save_processing_report(self, output_dir: Path):
        """Save detailed processing report."""
        report_path = output_dir / 'preprocessing_report.yaml'
        
        report = {
            'preprocessing_config': self.config,
            'statistics': self.processing_stats,
            'summary': {
                'success_rate': (
                    self.processing_stats['successful_files'] / 
                    max(self.processing_stats['total_files'], 1)
                ),
                'total_data_hours': self.processing_stats['total_duration_hours'],
                'avg_artifacts_ratio': (
                    self.processing_stats['artifacts_detected'] / 
                    max(self.processing_stats['successful_files'], 1)
                ),
            }
        }
        
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        self.logger.info(f"Processing report saved to {report_path}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Neural Data Preprocessing Pipeline")
    parser.add_argument("--input", type=str, required=True,
                      help="Input directory containing neural data files")
    parser.add_argument("--output", type=str, required=True,
                      help="Output directory for processed data")
    parser.add_argument("--config", type=str, default=None,
                      help="Path to preprocessing configuration file")
    parser.add_argument("--workers", type=int, default=None,
                      help="Number of parallel workers")
    parser.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
    parser.add_argument("--dry-run", action="store_true",
                      help="Perform dry run without actual processing")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Validate paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        return 1
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Number of workers: {args.workers}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be processed")
        
        # Just scan for files
        supported_extensions = ['.fif', '.fiff', '.h5', '.edf', '.bdf', '.mat']
        input_files = []
        
        for ext in supported_extensions:
            input_files.extend(input_dir.glob(f'*{ext}'))
            input_files.extend(input_dir.glob(f'**/*{ext}'))
        
        logger.info(f"Found {len(input_files)} files to process:")
        for file_path in sorted(input_files):
            logger.info(f"  - {file_path}")
        
        return 0
    
    try:
        # Initialize preprocessor
        preprocessor = NeuralDataPreprocessor(config_path=args.config)
        
        # Process directory
        stats = preprocessor.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            num_workers=args.workers
        )
        
        # Print final statistics
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total files: {stats['total_files']}")
        logger.info(f"Successful: {stats['successful_files']}")
        logger.info(f"Failed: {stats['failed_files']}")
        logger.info(f"Success rate: {stats['successful_files']/max(stats['total_files'], 1):.1%}")
        logger.info(f"Total data processed: {stats['total_duration_hours']:.1f} hours")
        logger.info(f"Output directory: {output_dir}")
        
        if stats['failed_files'] > 0:
            logger.warning(f"{stats['failed_files']} files failed processing")
            return 1
        else:
            logger.info("All files processed successfully!")
            return 0
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())