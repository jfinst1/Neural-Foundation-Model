"""
Streaming Neural Data Loader

Memory-efficient data loader for continuous neural recordings.
Handles temporal continuity, artifact detection, and real-time processing.
"""

import asyncio
import logging
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import mne
from scipy import signal
from scipy.stats import zscore

from ..signal_processing.filters import CausalButterworthFilter
from ..signal_processing.artifacts import ArtifactDetector
from ..privacy.anonymization import NeuralDataAnonymizer


class NeuralSession:
    """
    Represents a single neural recording session.
    """
    
    def __init__(
        self,
        session_path: Union[str, Path],
        channels: Optional[List[str]] = None,
        sampling_rate: Optional[int] = None
    ):
        self.session_path = Path(session_path)
        self.channels = channels
        self.sampling_rate = sampling_rate
        self._data_handle = None
        self._metadata = None
        
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def open(self):
        """Open the data file and load metadata."""
        if self.session_path.suffix == '.h5':
            self._data_handle = h5py.File(self.session_path, 'r')
            self._load_h5_metadata()
        elif self.session_path.suffix in ['.fif', '.fiff']:
            self._data_handle = mne.io.read_raw_fif(str(self.session_path), preload=False)
            self._load_mne_metadata()
        else:
            raise ValueError(f"Unsupported file format: {self.session_path.suffix}")
    
    def close(self):
        """Close the data file."""
        if self._data_handle is not None:
            if hasattr(self._data_handle, 'close'):
                self._data_handle.close()
            self._data_handle = None
    
    def _load_h5_metadata(self):
        """Load metadata from HDF5 file."""
        self._metadata = {
            'sampling_rate': self._data_handle.attrs.get('sampling_rate', 1000),
            'num_channels': self._data_handle['data'].shape[0],
            'duration_samples': self._data_handle['data'].shape[1],
            'channel_names': self._data_handle.attrs.get('channel_names', []),
            'subject_id': self._data_handle.attrs.get('subject_id', 'unknown')
        }
        
        if self.sampling_rate is None:
            self.sampling_rate = self._metadata['sampling_rate']
    
    def _load_mne_metadata(self):
        """Load metadata from MNE Raw object."""
        self._metadata = {
            'sampling_rate': int(self._data_handle.info['sfreq']),
            'num_channels': len(self._data_handle.ch_names),
            'duration_samples': len(self._data_handle.times),
            'channel_names': self._data_handle.ch_names,
            'subject_id': self._data_handle.info.get('subject_info', {}).get('id', 'unknown')
        }
        
        if self.sampling_rate is None:
            self.sampling_rate = self._metadata['sampling_rate']
    
    def read_data(
        self, 
        start_sample: int, 
        end_sample: int, 
        channels: Optional[List[int]] = None
    ) -> np.ndarray:
        """Read a chunk of neural data."""
        if self._data_handle is None:
            raise RuntimeError("Session not opened. Use context manager or call open().")
        
        if isinstance(self._data_handle, h5py.File):
            data = self._data_handle['data']
            if channels is not None:
                data = data[channels, start_sample:end_sample]
            else:
                data = data[:, start_sample:end_sample]
            return np.array(data)
        
        elif hasattr(self._data_handle, 'get_data'):  # MNE Raw object
            if channels is not None:
                picks = channels
            else:
                picks = 'all'
            
            data, _ = self._data_handle.get_data(
                picks=picks,
                start=start_sample,
                stop=end_sample,
                return_times=False
            )
            return data
        
        else:
            raise RuntimeError("Unknown data handle type")
    
    @property
    def metadata(self) -> Dict:
        """Get session metadata."""
        if self._metadata is None:
            raise RuntimeError("Session not opened")
        return self._metadata


class TemporalWindow:
    """
    Represents a temporal window of neural data with overlap handling.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        start_time: float,
        end_time: float,
        sampling_rate: int,
        session_id: str,
        subject_id: str,
        overlap_samples: int = 0
    ):
        self.data = data  # Shape: (channels, time)
        self.start_time = start_time
        self.end_time = end_time
        self.sampling_rate = sampling_rate
        self.session_id = session_id
        self.subject_id = subject_id
        self.overlap_samples = overlap_samples
        
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def num_samples(self) -> int:
        """Number of time samples."""
        return self.data.shape[1]
    
    @property
    def num_channels(self) -> int:
        """Number of channels."""
        return self.data.shape[0]
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor."""
        return torch.from_numpy(self.data).float()


class StreamingNeuralDataset(IterableDataset):
    """
    Streaming dataset for neural data that handles temporal continuity.
    """
    
    def __init__(
        self,
        session_paths: List[Union[str, Path]],
        temporal_window_size: float = 10.0,  # seconds
        overlap_ratio: float = 0.1,
        sampling_rate: int = 1000,
        channels: Optional[List[int]] = None,
        preprocessing_config: Optional[Dict] = None,
        artifact_detection: bool = True,
        privacy_config: Optional[Dict] = None,
        max_workers: int = 4
    ):
        self.session_paths = [Path(p) for p in session_paths]
        self.temporal_window_size = temporal_window_size
        self.overlap_ratio = overlap_ratio
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.preprocessing_config = preprocessing_config or {}
        self.artifact_detection = artifact_detection
        self.privacy_config = privacy_config or {}
        self.max_workers = max_workers
        
        # Calculate window parameters
        self.window_samples = int(temporal_window_size * sampling_rate)
        self.overlap_samples = int(self.window_samples * overlap_ratio)
        self.step_samples = self.window_samples - self.overlap_samples
        
        # Initialize processors
        self._init_processors()
        
        # Session metadata
        self.session_metadata = []
        self._load_session_metadata()
        
        self.logger = logging.getLogger(__name__)
    
    def _init_processors(self):
        """Initialize signal processing components."""
        # Signal filters
        if self.preprocessing_config.get('enable_filtering', True):
            self.bandpass_filter = CausalButterworthFilter(
                low_freq=self.preprocessing_config.get('low_freq', 1.0),
                high_freq=self.preprocessing_config.get('high_freq', 100.0),
                sampling_rate=self.sampling_rate,
                order=4
            )
        else:
            self.bandpass_filter = None
        
        # Artifact detection
        if self.artifact_detection:
            self.artifact_detector = ArtifactDetector(
                sampling_rate=self.sampling_rate,
                channels=self.channels
            )
        else:
            self.artifact_detector = None
        
        # Privacy protection
        if self.privacy_config.get('enable_anonymization', False):
            self.anonymizer = NeuralDataAnonymizer(
                method=self.privacy_config.get('method', 'differential_privacy'),
                noise_level=self.privacy_config.get('noise_level', 0.1)
            )
        else:
            self.anonymizer = None
    
    def _load_session_metadata(self):
        """Load metadata for all sessions."""
        for session_path in self.session_paths:
            try:
                with NeuralSession(session_path, sampling_rate=self.sampling_rate) as session:
                    metadata = session.metadata.copy()
                    metadata['session_path'] = session_path
                    metadata['num_windows'] = self._calculate_num_windows(
                        metadata['duration_samples']
                    )
                    self.session_metadata.append(metadata)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata for {session_path}: {e}")
    
    def _calculate_num_windows(self, total_samples: int) -> int:
        """Calculate number of temporal windows in a session."""
        if total_samples < self.window_samples:
            return 0
        return (total_samples - self.window_samples) // self.step_samples + 1
    
    def __iter__(self) -> Iterator[TemporalWindow]:
        """Iterate over temporal windows."""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single process
            session_indices = list(range(len(self.session_metadata)))
        else:
            # Multi-process: split sessions across workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            session_indices = list(range(worker_id, len(self.session_metadata), num_workers))
        
        for session_idx in session_indices:
            yield from self._iterate_session(session_idx)
    
    def _iterate_session(self, session_idx: int) -> Iterator[TemporalWindow]:
        """Iterate over windows in a single session."""
        session_meta = self.session_metadata[session_idx]
        session_path = session_meta['session_path']
        
        try:
            with NeuralSession(session_path, sampling_rate=self.sampling_rate) as session:
                total_samples = session_meta['duration_samples']
                num_windows = session_meta['num_windows']
                
                for window_idx in range(num_windows):
                    start_sample = window_idx * self.step_samples
                    end_sample = start_sample + self.window_samples
                    
                    # Ensure we don't go beyond session boundaries
                    if end_sample > total_samples:
                        end_sample = total_samples
                        start_sample = max(0, end_sample - self.window_samples)
                    
                    # Read raw data
                    raw_data = session.read_data(
                        start_sample=start_sample,
                        end_sample=end_sample,
                        channels=self.channels
                    )
                    
                    # Apply preprocessing
                    processed_data = self._preprocess_window(raw_data)
                    
                    # Skip if artifacts detected
                    if self._should_skip_window(processed_data):
                        continue
                    
                    # Create temporal window
                    start_time = start_sample / self.sampling_rate
                    end_time = end_sample / self.sampling_rate
                    
                    window = TemporalWindow(
                        data=processed_data,
                        start_time=start_time,
                        end_time=end_time,
                        sampling_rate=self.sampling_rate,
                        session_id=str(session_path.stem),
                        subject_id=session_meta['subject_id'],
                        overlap_samples=self.overlap_samples
                    )
                    
                    yield window
                    
        except Exception as e:
            self.logger.error(f"Error processing session {session_path}: {e}")
    
    def _preprocess_window(self, data: np.ndarray) -> np.ndarray:
        """Apply preprocessing to a data window."""
        processed = data.copy()
        
        # Apply bandpass filtering
        if self.bandpass_filter is not None:
            processed = self.bandpass_filter.apply(processed)
        
        # Z-score normalization
        if self.preprocessing_config.get('normalize', True):
            processed = zscore(processed, axis=1, nan_policy='omit')
        
        # Apply privacy protection
        if self.anonymizer is not None:
            processed = self.anonymizer.anonymize(processed)
        
        return processed
    
    def _should_skip_window(self, data: np.ndarray) -> bool:
        """Check if window should be skipped due to artifacts."""
        if self.artifact_detector is None:
            return False
        
        return self.artifact_detector.detect_artifacts(data)


class RealTimeNeuralDataLoader:
    """
    Real-time data loader for streaming neural data acquisition.
    """
    
    def __init__(
        self,
        acquisition_interface: 'AcquisitionInterface',
        buffer_size: int = 10000,  # samples
        sampling_rate: int = 1000,
        channels: Optional[List[int]] = None,
        preprocessing_config: Optional[Dict] = None
    ):
        self.acquisition_interface = acquisition_interface
        self.buffer_size = buffer_size
        self.sampling_rate = sampling_rate
        self.channels = channels or list(range(acquisition_interface.num_channels))
        self.preprocessing_config = preprocessing_config or {}
        
        # Circular buffer for streaming data
        self.buffer = CircularBuffer(
            size=buffer_size,
            num_channels=len(self.channels)
        )
        
        # Processing components
        self._init_processors()
        
        # State
        self.is_streaming = False
        self._streaming_task = None
        self.logger = logging.getLogger(__name__)
    
    def _init_processors(self):
        """Initialize real-time signal processing."""
        if self.preprocessing_config.get('enable_filtering', True):
            self.filter = CausalButterworthFilter(
                low_freq=self.preprocessing_config.get('low_freq', 1.0),
                high_freq=self.preprocessing_config.get('high_freq', 100.0),
                sampling_rate=self.sampling_rate,
                order=4
            )
        else:
            self.filter = None
        
        self.artifact_detector = ArtifactDetector(
            sampling_rate=self.sampling_rate,
            channels=self.channels,
            real_time=True
        )
    
    async def start_streaming(self):
        """Start real-time data streaming."""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self._streaming_task = asyncio.create_task(self._streaming_loop())
        self.logger.info("Started real-time neural data streaming")
    
    async def stop_streaming(self):
        """Stop real-time data streaming."""
        self.is_streaming = False
        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped real-time neural data streaming")
    
    async def _streaming_loop(self):
        """Main streaming loop."""
        try:
            async for sample in self.acquisition_interface.stream():
                if not self.is_streaming:
                    break
                
                # Select channels
                channel_data = sample[self.channels] if self.channels else sample
                
                # Apply real-time preprocessing
                processed_sample = await self._process_sample(channel_data)
                
                # Add to buffer
                self.buffer.append(processed_sample)
                
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            raise
    
    async def _process_sample(self, sample: np.ndarray) -> np.ndarray:
        """Process a single sample in real-time."""
        processed = sample.copy()
        
        # Apply causal filtering
        if self.filter is not None:
            processed = self.filter.apply_sample(processed)
        
        return processed
    
    def get_recent_data(self, duration: float) -> np.ndarray:
        """Get recent data from buffer."""
        num_samples = int(duration * self.sampling_rate)
        return self.buffer.get_recent(num_samples)
    
    async def stream_windows(
        self, 
        window_size: float = 1.0,
        overlap_ratio: float = 0.5
    ) -> Iterator[np.ndarray]:
        """Stream overlapping time windows."""
        window_samples = int(window_size * self.sampling_rate)
        step_samples = int(window_samples * (1 - overlap_ratio))
        
        last_window_start = 0
        
        while self.is_streaming:
            current_buffer_size = self.buffer.current_size()
            
            if current_buffer_size >= window_samples:
                window_start = current_buffer_size - window_samples
                
                # Check if we have enough new data for next window
                if window_start >= last_window_start + step_samples:
                    window_data = self.buffer.get_range(window_start, window_samples)
                    
                    # Skip if artifacts detected
                    if not self.artifact_detector.detect_artifacts(window_data):
                        yield window_data
                    
                    last_window_start = window_start
            
            # Wait before next check
            await asyncio.sleep(0.001)  # 1ms


class CircularBuffer:
    """
    Efficient circular buffer for streaming neural data.
    """
    
    def __init__(self, size: int, num_channels: int):
        self.size = size
        self.num_channels = num_channels
        self.buffer = np.zeros((num_channels, size), dtype=np.float32)
        self.write_ptr = 0
        self.is_full = False
        self._lock = asyncio.Lock()
    
    async def append(self, data: np.ndarray):
        """Append new data to buffer."""
        async with self._lock:
            if data.ndim == 1:
                # Single sample
                self.buffer[:, self.write_ptr] = data
                self.write_ptr = (self.write_ptr + 1) % self.size
                if self.write_ptr == 0:
                    self.is_full = True
            else:
                # Multiple samples
                num_samples = data.shape[1]
                for i in range(num_samples):
                    self.buffer[:, self.write_ptr] = data[:, i]
                    self.write_ptr = (self.write_ptr + 1) % self.size
                    if self.write_ptr == 0:
                        self.is_full = True
    
    def get_recent(self, num_samples: int) -> np.ndarray:
        """Get most recent samples."""
        if not self.is_full and num_samples > self.write_ptr:
            num_samples = self.write_ptr
        
        if num_samples > self.size:
            num_samples = self.size
        
        if self.is_full:
            # Buffer is full, data wraps around
            start_idx = (self.write_ptr - num_samples) % self.size
            if start_idx + num_samples <= self.size:
                return self.buffer[:, start_idx:start_idx + num_samples]
            else:
                # Wrap around
                part1 = self.buffer[:, start_idx:]
                part2 = self.buffer[:, :start_idx + num_samples - self.size]
                return np.concatenate([part1, part2], axis=1)
        else:
            # Buffer not full yet
            start_idx = max(0, self.write_ptr - num_samples)
            return self.buffer[:, start_idx:self.write_ptr]
    
    def get_range(self, start_offset: int, num_samples: int) -> np.ndarray:
        """Get data range from buffer."""
        if not self.is_full and start_offset + num_samples > self.write_ptr:
            raise ValueError("Requested range beyond available data")
        
        current_size = self.current_size()
        actual_start = current_size - start_offset - num_samples
        
        if actual_start < 0:
            raise ValueError("Requested range beyond buffer capacity")
        
        return self.get_recent(start_offset + num_samples)[:, :num_samples]
    
    def current_size(self) -> int:
        """Get current buffer size."""
        return self.size if self.is_full else self.write_ptr


class NeuralDataLoader:
    """
    High-level interface for neural data loading.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path, List[Path]],
        batch_size: int = 1,
        temporal_window: float = 10.0,
        sampling_rate: int = 1000,
        channels: Optional[List[int]] = None,
        preprocessing_config: Optional[Dict] = None,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.temporal_window = temporal_window
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.preprocessing_config = preprocessing_config or {}
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Resolve paths
        if isinstance(data_path, (str, Path)):
            data_path = Path(data_path)
            if data_path.is_dir():
                self.session_paths = list(data_path.glob("*.h5")) + list(data_path.glob("*.fif"))
            else:
                self.session_paths = [data_path]
        else:
            self.session_paths = [Path(p) for p in data_path]
        
        # Create dataset
        self.dataset = StreamingNeuralDataset(
            session_paths=self.session_paths,
            temporal_window_size=temporal_window,
            sampling_rate=sampling_rate,
            channels=channels,
            preprocessing_config=preprocessing_config
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized NeuralDataLoader with {len(self.session_paths)} sessions")
    
    def __iter__(self):
        """Create data loader iterator."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_temporal_windows
        ).__iter__()
    
    def _collate_temporal_windows(self, batch: List[TemporalWindow]) -> Dict[str, torch.Tensor]:
        """Collate function for temporal windows."""
        # Stack neural data
        neural_data = torch.stack([window.to_tensor() for window in batch])
        
        # Create metadata tensors
        subject_ids = [window.subject_id for window in batch]
        session_ids = [window.session_id for window in batch]
        start_times = torch.tensor([window.start_time for window in batch])
        
        return {
            'neural_data': neural_data,
            'subject_ids': subject_ids,
            'session_ids': session_ids,
            'start_times': start_times,
            'metadata': {
                'sampling_rate': self.sampling_rate,
                'temporal_window': self.temporal_window
            }
        }
    
    def stream(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream neural data windows."""
        for batch in self:
            yield batch
    
    def get_session_info(self) -> List[Dict]:
        """Get information about all sessions."""
        return self.dataset.session_metadata


# Mock acquisition interface for testing
class MockAcquisitionInterface:
    """Mock neural data acquisition interface for testing."""
    
    def __init__(self, num_channels: int = 64, sampling_rate: int = 1000):
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self._rng = np.random.RandomState(42)
    
    async def stream(self):
        """Generate mock neural data stream."""
        t = 0
        dt = 1.0 / self.sampling_rate
        
        while True:
            # Generate realistic neural-like signal
            # Mix of different frequency components + noise
            low_freq = 0.1 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha-like
            high_freq = 0.05 * np.sin(2 * np.pi * 40 * t)  # 40 Hz gamma-like  
            noise = 0.02 * self._rng.randn(self.num_channels)
            
            # Add some spatial correlation between channels
            base_signal = low_freq + high_freq
            spatial_pattern = np.exp(-np.arange(self.num_channels) / 10.0)
            
            sample = base_signal * spatial_pattern + noise
            
            yield sample
            
            t += dt
            await asyncio.sleep(dt)  # Simulate real-time acquisition


# Example usage and testing
async def test_streaming_loader():
    """Test the streaming neural data loader."""
    # Create mock acquisition interface
    acquisition = MockAcquisitionInterface(num_channels=64, sampling_rate=1000)
    
    # Create real-time loader
    real_time_loader = RealTimeNeuralDataLoader(
        acquisition_interface=acquisition,
        buffer_size=10000,
        sampling_rate=1000,
        preprocessing_config={
            'enable_filtering': True,
            'low_freq': 1.0,
            'high_freq': 100.0,
            'normalize': True
        }
    )
    
    # Start streaming
    await real_time_loader.start_streaming()
    
    # Stream windows
    window_count = 0
    async for window in real_time_loader.stream_windows(window_size=1.0, overlap_ratio=0.5):
        print(f"Received window {window_count}: shape {window.shape}")
        window_count += 1
        
        if window_count >= 10:  # Stop after 10 windows
            break
    
    # Stop streaming
    await real_time_loader.stop_streaming()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with mock data
    asyncio.run(test_streaming_loader())