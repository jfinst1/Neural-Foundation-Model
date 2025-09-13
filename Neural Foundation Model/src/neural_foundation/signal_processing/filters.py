"""
Real-time Signal Processing Filters for Neural Data

Causal filters optimized for streaming neural data processing with
minimal latency and phase distortion.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from typing import Optional, Tuple, Union
import warnings


class CausalButterworthFilter:
    """
    Causal Butterworth filter for real-time neural signal processing.
    Maintains filter state for streaming data.
    """
    
    def __init__(
        self,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
        sampling_rate: float = 1000.0,
        order: int = 4,
        filter_type: str = 'bandpass'
    ):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.sampling_rate = sampling_rate
        self.order = order
        self.filter_type = filter_type
        
        # Design filter coefficients
        self._design_filter()
        
        # Initialize filter states (for streaming)
        self.filter_states = {}
        
    def _design_filter(self):
        """Design the Butterworth filter coefficients."""
        nyquist = self.sampling_rate / 2.0
        
        if self.filter_type == 'lowpass':
            if self.high_freq is None:
                raise ValueError("high_freq required for lowpass filter")
            critical_freq = self.high_freq / nyquist
            self.b, self.a = signal.butter(
                self.order, critical_freq, btype='low', analog=False
            )
            
        elif self.filter_type == 'highpass':
            if self.low_freq is None:
                raise ValueError("low_freq required for highpass filter")
            critical_freq = self.low_freq / nyquist
            self.b, self.a = signal.butter(
                self.order, critical_freq, btype='high', analog=False
            )
            
        elif self.filter_type == 'bandpass':
            if self.low_freq is None or self.high_freq is None:
                raise ValueError("Both low_freq and high_freq required for bandpass filter")
            low_critical = self.low_freq / nyquist
            high_critical = self.high_freq / nyquist
            self.b, self.a = signal.butter(
                self.order, [low_critical, high_critical], btype='band', analog=False
            )
            
        elif self.filter_type == 'bandstop':
            if self.low_freq is None or self.high_freq is None:
                raise ValueError("Both low_freq and high_freq required for bandstop filter")
            low_critical = self.low_freq / nyquist
            high_critical = self.high_freq / nyquist
            self.b, self.a = signal.butter(
                self.order, [low_critical, high_critical], btype='bandstop', analog=False
            )
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
    
    def apply(self, data: np.ndarray, channel_id: Optional[int] = None) -> np.ndarray:
        """
        Apply filter to data array.
        
        Args:
            data: Input data (channels, time) or (time,) for single channel
            channel_id: Channel ID for state tracking (required for streaming)
            
        Returns:
            Filtered data with same shape as input
        """
        if data.ndim == 1:
            # Single channel
            return self._filter_single_channel(data, channel_id or 0)
        elif data.ndim == 2:
            # Multi-channel: (channels, time)
            filtered = np.zeros_like(data)
            for ch_idx in range(data.shape[0]):
                filtered[ch_idx] = self._filter_single_channel(
                    data[ch_idx], channel_id=ch_idx
                )
            return filtered
        else:
            raise ValueError(f"Data must be 1D or 2D, got shape {data.shape}")
    
    def _filter_single_channel(self, data: np.ndarray, channel_id: int) -> np.ndarray:
        """Filter single channel with state preservation."""
        if channel_id not in self.filter_states:
            # Initialize filter state
            self.filter_states[channel_id] = signal.lfilter_zi(self.b, self.a) * data[0]
        
        # Apply filter with initial conditions
        filtered_data, self.filter_states[channel_id] = signal.lfilter(
            self.b, self.a, data, zi=self.filter_states[channel_id]
        )
        
        return filtered_data
    
    def apply_sample(self, sample: Union[float, np.ndarray], channel_id: int = 0) -> Union[float, np.ndarray]:
        """
        Apply filter to single sample (for real-time processing).
        
        Args:
            sample: Single sample or array of samples (one per channel)
            channel_id: Channel ID for single sample, ignored for array
            
        Returns:
            Filtered sample(s)
        """
        if isinstance(sample, (int, float)):
            return self._filter_single_sample(sample, channel_id)
        else:
            # Array of samples (one per channel)
            filtered = np.zeros_like(sample)
            for ch_idx, ch_sample in enumerate(sample):
                filtered[ch_idx] = self._filter_single_sample(ch_sample, ch_idx)
            return filtered
    
    def _filter_single_sample(self, sample: float, channel_id: int) -> float:
        """Filter single sample with state update."""
        if channel_id not in self.filter_states:
            # Initialize state with current sample
            self.filter_states[channel_id] = np.full(
                len(self.a) - 1, sample, dtype=np.float64
            )
        
        # Manual filter implementation for single sample
        state = self.filter_states[channel_id]
        
        # Apply filter equation: y[n] = b[0]*x[n] + state[0]
        output = self.b[0] * sample + state[0]
        
        # Update state
        for i in range(len(state) - 1):
            state[i] = (self.b[i + 1] * sample - self.a[i + 1] * output + state[i + 1])
        
        if len(state) > 0:
            state[-1] = self.b[-1] * sample - self.a[-1] * output
        
        return output
    
    def reset_states(self, channel_ids: Optional[list] = None):
        """Reset filter states for specified channels or all channels."""
        if channel_ids is None:
            self.filter_states.clear()
        else:
            for ch_id in channel_ids:
                if ch_id in self.filter_states:
                    del self.filter_states[ch_id]


class NotchFilter:
    """
    Notch filter for removing power line interference (50Hz/60Hz).
    """
    
    def __init__(
        self,
        notch_freq: float = 50.0,
        quality_factor: float = 30.0,
        sampling_rate: float = 1000.0
    ):
        self.notch_freq = notch_freq
        self.quality_factor = quality_factor
        self.sampling_rate = sampling_rate
        
        # Design notch filter
        self._design_filter()
        self.filter_states = {}
    
    def _design_filter(self):
        """Design the notch filter coefficients."""
        nyquist = self.sampling_rate / 2.0
        freq_normalized = self.notch_freq / nyquist
        
        self.b, self.a = signal.iirnotch(
            freq_normalized, self.quality_factor, fs=self.sampling_rate
        )
    
    def apply(self, data: np.ndarray, channel_id: Optional[int] = None) -> np.ndarray:
        """Apply notch filter to data."""
        if data.ndim == 1:
            return self._filter_single_channel(data, channel_id or 0)
        elif data.ndim == 2:
            filtered = np.zeros_like(data)
            for ch_idx in range(data.shape[0]):
                filtered[ch_idx] = self._filter_single_channel(
                    data[ch_idx], channel_id=ch_idx
                )
            return filtered
        else:
            raise ValueError(f"Data must be 1D or 2D, got shape {data.shape}")
    
    def _filter_single_channel(self, data: np.ndarray, channel_id: int) -> np.ndarray:
        """Filter single channel with state preservation."""
        if channel_id not in self.filter_states:
            self.filter_states[channel_id] = signal.lfilter_zi(self.b, self.a) * data[0]
        
        filtered_data, self.filter_states[channel_id] = signal.lfilter(
            self.b, self.a, data, zi=self.filter_states[channel_id]
        )
        
        return filtered_data


class AdaptiveFilter:
    """
    Adaptive filter for removing artifacts and noise.
    Uses LMS (Least Mean Squares) algorithm.
    """
    
    def __init__(
        self,
        filter_length: int = 32,
        step_size: float = 0.01,
        leak_factor: float = 0.99
    ):
        self.filter_length = filter_length
        self.step_size = step_size
        self.leak_factor = leak_factor
        
        # Filter weights and buffers
        self.weights = {}  # Per channel
        self.input_buffer = {}  # Per channel
        
    def apply(self, primary_input: np.ndarray, reference_input: np.ndarray, 
              channel_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply adaptive filter to remove artifacts.
        
        Args:
            primary_input: Primary signal (contains desired signal + noise)
            reference_input: Reference signal (noise/artifact)
            channel_id: Channel identifier
            
        Returns:
            Tuple of (filtered_signal, estimated_noise)
        """
        if channel_id not in self.weights:
            self.weights[channel_id] = np.zeros(self.filter_length)
            self.input_buffer[channel_id] = np.zeros(self.filter_length)
        
        weights = self.weights[channel_id]
        buffer = self.input_buffer[channel_id]
        
        filtered_signal = np.zeros_like(primary_input)
        estimated_noise = np.zeros_like(primary_input)
        
        for i in range(len(primary_input)):
            # Update input buffer
            buffer[1:] = buffer[:-1]
            buffer[0] = reference_input[i]
            
            # Estimate noise
            estimated_noise[i] = np.dot(weights, buffer)
            
            # Compute error (desired signal)
            error = primary_input[i] - estimated_noise[i]
            filtered_signal[i] = error
            
            # Update weights using LMS algorithm
            weights = self.leak_factor * weights + \
                      2 * self.step_size * error * buffer
        
        self.weights[channel_id] = weights
        self.input_buffer[channel_id] = buffer
        
        return filtered_signal, estimated_noise


class CUDASignalProcessor(nn.Module):
    """
    CUDA-accelerated signal processing for real-time applications.
    """
    
    def __init__(
        self,
        num_channels: int = 64,
        sampling_rate: float = 1000.0,
        filter_configs: dict = None
    ):
        super().__init__()
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        
        # Default filter configurations
        if filter_configs is None:
            filter_configs = {
                'bandpass': {'low_freq': 1.0, 'high_freq': 100.0, 'order': 4},
                'notch': {'freq': 50.0, 'quality': 30.0}
            }
        
        # Create learnable filter parameters
        self._create_filters(filter_configs)
        
        # Circular buffers for streaming
        self.buffer_size = 1000
        self.register_buffer(
            'input_buffer',
            torch.zeros(num_channels, self.buffer_size)
        )
        self.register_buffer('buffer_ptr', torch.zeros(1, dtype=torch.long))
        
    def _create_filters(self, filter_configs: dict):
        """Create learnable filter parameters."""
        # Bandpass filter coefficients as learnable parameters
        if 'bandpass' in filter_configs:
            bp_config = filter_configs['bandpass']
            # Convert scipy filter to PyTorch parameters
            b, a = self._design_bandpass_filter(**bp_config)
            self.register_buffer('bandpass_b', torch.from_numpy(b).float())
            self.register_buffer('bandpass_a', torch.from_numpy(a).float())
        
        # Notch filter coefficients
        if 'notch' in filter_configs:
            notch_config = filter_configs['notch']
            b, a = self._design_notch_filter(**notch_config)
            self.register_buffer('notch_b', torch.from_numpy(b).float())
            self.register_buffer('notch_a', torch.from_numpy(a).float())
    
    def _design_bandpass_filter(self, low_freq: float, high_freq: float, order: int):
        """Design bandpass filter coefficients."""
        nyquist = self.sampling_rate / 2.0
        low_critical = low_freq / nyquist
        high_critical = high_freq / nyquist
        return signal.butter(order, [low_critical, high_critical], btype='band')
    
    def _design_notch_filter(self, freq: float, quality: float):
        """Design notch filter coefficients."""
        nyquist = self.sampling_rate / 2.0
        freq_normalized = freq / nyquist
        return signal.iirnotch(freq_normalized, quality, fs=self.sampling_rate)
    
    def filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply signal processing pipeline.
        
        Args:
            x: Input tensor (batch_size, channels, time) or (channels, time)
            
        Returns:
            Filtered tensor with same shape
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        batch_size, channels, time = x.shape
        
        # Apply bandpass filter
        if hasattr(self, 'bandpass_b'):
            x = self._apply_iir_filter(x, self.bandpass_b, self.bandpass_a)
        
        # Apply notch filter
        if hasattr(self, 'notch_b'):
            x = self._apply_iir_filter(x, self.notch_b, self.notch_a)
        
        return x.squeeze(0) if batch_size == 1 else x
    
    def _apply_iir_filter(self, x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Apply IIR filter using PyTorch operations."""
        batch_size, channels, time = x.shape
        
        # Normalize coefficients
        b = b / a[0]
        a = a / a[0]
        
        # Initialize output
        y = torch.zeros_like(x)
        
        # Apply filter equation: y[n] = sum(b[k]*x[n-k]) - sum(a[k]*y[n-k])
        for n in range(time):
            for k in range(len(b)):
                if n - k >= 0:
                    y[:, :, n] += b[k] * x[:, :, n - k]
            
            for k in range(1, len(a)):
                if n - k >= 0:
                    y[:, :, n] -= a[k] * y[:, :, n - k]
        
        return y
    
    def add_to_buffer(self, sample: torch.Tensor):
        """Add sample to circular buffer for streaming processing."""
        ptr = self.buffer_ptr.item()
        self.input_buffer[:, ptr] = sample
        self.buffer_ptr[0] = (ptr + 1) % self.buffer_size
    
    def get_recent_data(self, num_samples: int) -> torch.Tensor:
        """Get recent data from buffer."""
        ptr = self.buffer_ptr.item()
        if num_samples <= ptr:
            return self.input_buffer[:, ptr - num_samples:ptr]
        else:
            # Wrap around buffer
            part1 = self.input_buffer[:, self.buffer_size - (num_samples - ptr):]
            part2 = self.input_buffer[:, :ptr]
            return torch.cat([part1, part2], dim=1)


class MultiChannelCAR:
    """
    Common Average Reference (CAR) filter for multi-channel neural data.
    Removes common-mode artifacts across channels.
    """
    
    def __init__(self, exclude_channels: Optional[list] = None):
        self.exclude_channels = exclude_channels or []
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply CAR filtering.
        
        Args:
            data: Input data (channels, time)
            
        Returns:
            CAR-filtered data
        """
        if data.ndim != 2:
            raise ValueError("Data must be 2D (channels, time)")
        
        # Get channels to use for reference
        all_channels = set(range(data.shape[0]))
        ref_channels = list(all_channels - set(self.exclude_channels))
        
        if len(ref_channels) == 0:
            warnings.warn("No channels available for CAR reference")
            return data
        
        # Compute common average
        common_average = np.mean(data[ref_channels], axis=0)
        
        # Subtract from all channels
        car_data = data - common_average[np.newaxis, :]
        
        return car_data


class SpatialFilter:
    """
    Spatial filtering for neural data (Laplacian, bipolar, etc.).
    """
    
    def __init__(self, filter_type: str = 'laplacian', channel_positions: Optional[np.ndarray] = None):
        self.filter_type = filter_type
        self.channel_positions = channel_positions
        self.spatial_weights = None
        
        if filter_type == 'laplacian' and channel_positions is not None:
            self._compute_laplacian_weights()
    
    def _compute_laplacian_weights(self):
        """Compute Laplacian filter weights based on channel positions."""
        num_channels = len(self.channel_positions)
        self.spatial_weights = np.zeros((num_channels, num_channels))
        
        for i in range(num_channels):
            # Find neighboring channels
            distances = np.linalg.norm(
                self.channel_positions - self.channel_positions[i], axis=1
            )
            
            # Use channels within a certain radius as neighbors
            radius = np.median(distances[distances > 0]) * 2
            neighbors = np.where((distances > 0) & (distances <= radius))[0]
            
            if len(neighbors) > 0:
                # Set weights: center channel = 1, neighbors = -1/N
                self.spatial_weights[i, i] = 1.0
                self.spatial_weights[i, neighbors] = -1.0 / len(neighbors)
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply spatial filtering.
        
        Args:
            data: Input data (channels, time)
            
        Returns:
            Spatially filtered data
        """
        if self.filter_type == 'laplacian' and self.spatial_weights is not None:
            return np.dot(self.spatial_weights, data)
        elif self.filter_type == 'bipolar':
            return self._apply_bipolar_montage(data)
        else:
            warnings.warn(f"Spatial filter {self.filter_type} not implemented")
            return data
    
    def _apply_bipolar_montage(self, data: np.ndarray) -> np.ndarray:
        """Apply bipolar montage (adjacent channel differences)."""
        if data.shape[0] < 2:
            return data
        
        bipolar_data = np.zeros((data.shape[0] - 1, data.shape[1]))
        for i in range(data.shape[0] - 1):
            bipolar_data[i] = data[i] - data[i + 1]
        
        return bipolar_data