"""
Loss Functions for Neural Foundation Model Training

Specialized loss functions for neural data including temporal consistency,
subject invariance, contrastive learning, and reconstruction losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining different objectives for neural foundation model training.
    """
    
    def __init__(self, loss_components: Dict[str, nn.Module], adaptive_weighting: bool = True):
        super().__init__()
        self.loss_components = nn.ModuleDict(loss_components)
        self.adaptive_weighting = adaptive_weighting
        
        if adaptive_weighting:
            # Learnable loss weights (uncertainty-based weighting)
            self.log_vars = nn.Parameter(torch.zeros(len(loss_components)))
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            inputs: Dictionary containing predictions, targets, and metadata
            
        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        total_loss = 0.0
        
        # Compute individual losses
        for i, (loss_name, loss_fn) in enumerate(self.loss_components.items()):
            try:
                loss_value = loss_fn(inputs)
                losses[loss_name] = loss_value
                
                if self.adaptive_weighting:
                    # Uncertainty-based weighting
                    precision = torch.exp(-self.log_vars[i])
                    weighted_loss = precision * loss_value + self.log_vars[i]
                else:
                    # Use predefined weight from loss function
                    weight = getattr(loss_fn, 'weight', 1.0)
                    weighted_loss = weight * loss_value
                
                total_loss += weighted_loss
                
            except Exception as e:
                print(f"Warning: Loss {loss_name} computation failed: {e}")
                losses[loss_name] = torch.tensor(0.0, device=total_loss.device if hasattr(total_loss, 'device') else 'cpu')
        
        losses['total_loss'] = total_loss
        return losses


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for neural signals.
    """
    
    def __init__(
        self,
        loss_type: str = 'mse',
        weight: float = 1.0,
        reduction: str = 'mean',
        temporal_weighting: bool = False
    ):
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight
        self.reduction = reduction
        self.temporal_weighting = temporal_weighting
        
        # Loss functions
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'mae':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'huber':
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
        elif loss_type == 'spectral':
            self.loss_fn = self._spectral_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            inputs: Dictionary with 'predictions' and 'targets' keys
            
        Returns:
            Reconstruction loss tensor
        """
        predictions = inputs['predictions']['reconstruction']
        targets = inputs['targets']
        
        # Compute base loss
        if self.loss_type == 'spectral':
            loss = self.loss_fn(predictions, targets)
        else:
            loss = self.loss_fn(predictions, targets)
        
        # Apply temporal weighting if enabled
        if self.temporal_weighting:
            temporal_weights = self._compute_temporal_weights(targets)
            loss = loss * temporal_weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _spectral_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Spectral reconstruction loss comparing frequency domain representations.
        """
        # Compute FFT for both predictions and targets
        pred_fft = torch.fft.rfft(predictions, dim=-1)
        target_fft = torch.fft.rfft(targets, dim=-1)
        
        # Compute magnitude and phase losses
        mag_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft), reduction='none')
        phase_loss = F.mse_loss(torch.angle(pred_fft), torch.angle(target_fft), reduction='none')
        
        # Combine magnitude and phase losses
        spectral_loss = mag_loss + 0.1 * phase_loss  # Phase weighted less
        
        return spectral_loss
    
    def _compute_temporal_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal weights based on signal properties.
        Give higher weight to more informative time periods.
        """
        # Compute local variance as a proxy for informativeness
        window_size = min(50, targets.size(-1) // 10)
        padding = window_size // 2
        
        # Pad targets
        padded_targets = F.pad(targets, (padding, padding), mode='reflect')
        
        # Compute local variance using unfold
        unfolded = padded_targets.unfold(-1, window_size, 1)
        local_variance = torch.var(unfolded, dim=-1)
        
        # Normalize weights
        weights = local_variance / (local_variance.mean(dim=-1, keepdim=True) + 1e-8)
        weights = torch.clamp(weights, 0.5, 2.0)  # Limit weight range
        
        return weights.unsqueeze(-2)  # Add channel dimension


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning neural representations.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        weight: float = 1.0,
        projection_dim: int = 128,
        method: str = 'simclr'
    ):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        self.projection_dim = projection_dim
        self.method = method
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            inputs: Dictionary with 'predictions' containing contrastive features
            
        Returns:
            Contrastive loss tensor
        """
        features = inputs['predictions']['contrastive']  # (batch, time, feature_dim)
        
        if self.method == 'simclr':
            return self._simclr_loss(features)
        elif self.method == 'temporal':
            return self._temporal_contrastive_loss(features)
        else:
            raise ValueError(f"Unknown contrastive method: {self.method}")
    
    def _simclr_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        SimCLR-style contrastive loss adapted for neural data.
        """
        batch_size, seq_len, feature_dim = features.shape
        
        # Create augmented pairs by temporal shifting
        shifted_features = torch.roll(features, shifts=seq_len//4, dims=1)
        
        # Flatten and concatenate
        features_flat = features.view(batch_size * seq_len, feature_dim)
        shifted_flat = shifted_features.view(batch_size * seq_len, feature_dim)
        
        # Combine positive pairs
        all_features = torch.cat([features_flat, shifted_flat], dim=0)
        
        # Normalize features
        all_features = F.normalize(all_features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(all_features, all_features.T) / self.temperature
        
        # Create positive pair labels
        batch_size_total = batch_size * seq_len
        labels = torch.arange(batch_size_total, device=features.device)
        labels = torch.cat([labels, labels])
        
        # Positive pairs: (i, i + batch_size_total)
        positive_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
        positive_mask[torch.arange(batch_size_total), torch.arange(batch_size_total, 2 * batch_size_total)] = True
        positive_mask[torch.arange(batch_size_total, 2 * batch_size_total), torch.arange(batch_size_total)] = True
        
        # Remove self-similarities
        identity_mask = torch.eye(2 * batch_size_total, device=features.device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(identity_mask, float('-inf'))
        
        # Compute contrastive loss
        positive_similarities = similarity_matrix[positive_mask].view(2 * batch_size_total, -1)
        negative_similarities = similarity_matrix[~positive_mask].view(2 * batch_size_total, -1)
        
        # InfoNCE loss
        logits = torch.cat([positive_similarities, negative_similarities], dim=1)
        labels = torch.zeros(2 * batch_size_total, dtype=torch.long, device=features.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def _temporal_contrastive_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        Temporal contrastive loss: nearby time points should be similar.
        """
        # Compute similarities between adjacent time points
        current_features = features[:, :-1, :]  # (batch, seq_len-1, feature_dim)
        next_features = features[:, 1:, :]      # (batch, seq_len-1, feature_dim)
        
        # Normalize features
        current_features = F.normalize(current_features, dim=-1)
        next_features = F.normalize(next_features, dim=-1)
        
        # Compute positive similarities (adjacent time points)
        positive_similarities = torch.sum(current_features * next_features, dim=-1)
        
        # Sample negative pairs (random time points from other sequences in batch)
        batch_size, seq_len, feature_dim = current_features.shape
        
        # Random negative samples
        neg_indices = torch.randint(0, batch_size * seq_len, (batch_size, seq_len, 8))
        neg_features = current_features.view(-1, feature_dim)[neg_indices.view(-1)]
        neg_features = neg_features.view(batch_size, seq_len, 8, feature_dim)
        
        # Compute negative similarities
        negative_similarities = torch.sum(
            current_features.unsqueeze(-2) * neg_features, dim=-1
        )
        
        # Contrastive loss: maximize positive, minimize negative
        positive_loss = -torch.log(torch.sigmoid(positive_similarities / self.temperature))
        negative_loss = -torch.log(torch.sigmoid(-negative_similarities / self.temperature))
        
        total_loss = positive_loss.mean() + negative_loss.mean()
        
        return total_loss


class TemporalConsistencyLoss(nn.Module):
    """
    Loss to encourage temporal consistency in neural representations.
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        consistency_type: str = 'smooth',
        temporal_window: int = 10
    ):
        super().__init__()
        self.weight = weight
        self.consistency_type = consistency_type
        self.temporal_window = temporal_window
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            inputs: Dictionary with 'predictions' containing features
            
        Returns:
            Temporal consistency loss
        """
        features = inputs['predictions']['features']  # (batch, time, feature_dim)
        
        if self.consistency_type == 'smooth':
            return self._smoothness_loss(features)
        elif self.consistency_type == 'predictive':
            return self._predictive_consistency_loss(features)
        else:
            raise ValueError(f"Unknown consistency type: {self.consistency_type}")
    
    def _smoothness_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encourage smooth temporal transitions.
        """
        # Compute first-order differences
        diff_1 = features[:, 1:, :] - features[:, :-1, :]
        smoothness_loss_1 = torch.mean(diff_1 ** 2)
        
        # Compute second-order differences (acceleration)
        diff_2 = diff_1[:, 1:, :] - diff_1[:, :-1, :]
        smoothness_loss_2 = torch.mean(diff_2 ** 2)
        
        return smoothness_loss_1 + 0.1 * smoothness_loss_2
    
    def _predictive_consistency_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encourage features to be predictable from past context.
        """
        batch_size, seq_len, feature_dim = features.shape
        
        if seq_len <= self.temporal_window:
            return torch.tensor(0.0, device=features.device)
        
        # Use past context to predict current features
        context_features = features[:, :-1, :]  # Past context
        target_features = features[:, 1:, :]    # Target to predict
        
        # Simple linear predictor (could be more sophisticated)
        if not hasattr(self, 'predictor'):
            self.predictor = nn.Linear(feature_dim, feature_dim).to(features.device)
        
        # Predict next features from current
        predicted_features = self.predictor(context_features)
        
        # Predictive loss
        prediction_loss = F.mse_loss(predicted_features, target_features)
        
        return prediction_loss


class SubjectInvarianceLoss(nn.Module):
    """
    Loss to encourage subject-invariant representations.
    Uses adversarial training to remove subject-specific information.
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        adversarial_alpha: float = 1.0,
        gradient_reversal: bool = True
    ):
        super().__init__()
        self.weight = weight
        self.adversarial_alpha = adversarial_alpha
        self.gradient_reversal = gradient_reversal
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute subject invariance loss.
        
        Args:
            inputs: Dictionary with 'predictions' and 'subject_ids'
            
        Returns:
            Subject invariance loss
        """
        if 'subject_logits' not in inputs['predictions']:
            return torch.tensor(0.0, device=list(inputs['predictions'].values())[0].device)
        
        subject_logits = inputs['predictions']['subject_logits']
        subject_ids = inputs['subject_ids']
        
        # Convert subject IDs to tensor if needed
        if not isinstance(subject_ids, torch.Tensor):
            subject_ids = torch.tensor(subject_ids, device=subject_logits.device)
        
        # Adversarial loss: maximize entropy of subject predictions
        # (makes it hard to predict subject from features)
        if self.gradient_reversal:
            # Gradient reversal: minimize subject classification accuracy
            subject_loss = F.cross_entropy(subject_logits, subject_ids)
            # Note: Gradient reversal is typically implemented in the model architecture
            return -self.adversarial_alpha * subject_loss
        else:
            # Standard adversarial loss
            subject_loss = F.cross_entropy(subject_logits, subject_ids)
            return self.adversarial_alpha * subject_loss


class SpectralLoss(nn.Module):
    """
    Loss based on frequency domain properties of neural signals.
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        freq_bands: Optional[List[Tuple[float, float]]] = None,
        sampling_rate: float = 1000.0
    ):
        super().__init__()
        self.weight = weight
        self.sampling_rate = sampling_rate
        
        # Default frequency bands of interest
        if freq_bands is None:
            freq_bands = [
                (1, 4),    # Delta
                (4, 8),    # Theta
                (8, 13),   # Alpha
                (13, 30),  # Beta
                (30, 100)  # Gamma
            ]
        
        self.freq_bands = freq_bands
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute spectral loss.
        
        Args:
            inputs: Dictionary with 'predictions' and 'targets'
            
        Returns:
            Spectral loss tensor
        """
        predictions = inputs['predictions']['reconstruction']
        targets = inputs['targets']
        
        # Compute power spectral densities
        pred_psd = self._compute_psd(predictions)
        target_psd = self._compute_psd(targets)
        
        # Compute loss for each frequency band
        total_loss = 0.0
        
        for low_freq, high_freq in self.freq_bands:
            # Get frequency indices for this band
            freq_indices = self._get_freq_indices(pred_psd.shape[-1], low_freq, high_freq)
            
            if len(freq_indices) > 0:
                pred_band = pred_psd[..., freq_indices]
                target_band = target_psd[..., freq_indices]
                
                band_loss = F.mse_loss(pred_band, target_band)
                total_loss += band_loss
        
        return total_loss / len(self.freq_bands)
    
    def _compute_psd(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Compute power spectral density using FFT.
        
        Args:
            signal: Input signal (batch, channels, time)
            
        Returns:
            Power spectral density (batch, channels, freq_bins)
        """
        # Apply window to reduce spectral leakage
        window = torch.hann_window(signal.shape[-1], device=signal.device)
        windowed_signal = signal * window
        
        # Compute FFT
        fft = torch.fft.rfft(windowed_signal, dim=-1)
        
        # Compute power spectral density
        psd = torch.abs(fft) ** 2
        
        return psd
    
    def _get_freq_indices(self, n_freq_bins: int, low_freq: float, high_freq: float) -> List[int]:
        """
        Get frequency bin indices for a given frequency range.
        """
        freq_resolution = self.sampling_rate / (2 * (n_freq_bins - 1))
        
        low_idx = int(low_freq / freq_resolution)
        high_idx = int(high_freq / freq_resolution)
        
        low_idx = max(0, low_idx)
        high_idx = min(n_freq_bins - 1, high_idx)
        
        return list(range(low_idx, high_idx + 1))


class CoherenceLoss(nn.Module):
    """
    Loss based on coherence between channels.
    Encourages realistic inter-channel relationships.
    """
    
    def __init__(self, weight: float = 1.0, reference_coherence: Optional[torch.Tensor] = None):
        super().__init__()
        self.weight = weight
        self.reference_coherence = reference_coherence
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute coherence loss.
        
        Args:
            inputs: Dictionary with 'predictions' and 'targets'
            
        Returns:
            Coherence loss tensor
        """
        predictions = inputs['predictions']['reconstruction']
        targets = inputs['targets']
        
        # Compute coherence for predictions and targets
        pred_coherence = self._compute_coherence(predictions)
        target_coherence = self._compute_coherence(targets)
        
        # Coherence loss
        coherence_loss = F.mse_loss(pred_coherence, target_coherence)
        
        return coherence_loss
    
    def _compute_coherence(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Compute coherence matrix between channels.
        
        Args:
            signal: Input signal (batch, channels, time)
            
        Returns:
            Coherence matrix (batch, channels, channels)
        """
        batch_size, n_channels, n_time = signal.shape
        
        # Compute cross-spectral densities
        fft = torch.fft.rfft(signal, dim=-1)  # (batch, channels, freq_bins)
        
        # Initialize coherence matrix
        coherence_matrix = torch.zeros(batch_size, n_channels, n_channels, device=signal.device)
        
        for i in range(n_channels):
            for j in range(n_channels):
                if i <= j:  # Only compute upper triangular + diagonal
                    # Cross-power spectral density
                    cross_psd = fft[:, i, :] * torch.conj(fft[:, j, :])
                    
                    # Auto-power spectral densities
                    auto_psd_i = torch.abs(fft[:, i, :]) ** 2
                    auto_psd_j = torch.abs(fft[:, j, :]) ** 2
                    
                    # Coherence
                    coherence = torch.abs(cross_psd) ** 2 / (auto_psd_i * auto_psd_j + 1e-8)
                    coherence = torch.mean(coherence, dim=-1)  # Average over frequencies
                    
                    coherence_matrix[:, i, j] = coherence
                    coherence_matrix[:, j, i] = coherence  # Symmetric matrix
        
        return coherence_matrix


class PhaseConsistencyLoss(nn.Module):
    """
    Loss to maintain phase relationships in neural oscillations.
    """
    
    def __init__(self, weight: float = 1.0, freq_bands: Optional[List[Tuple[float, float]]] = None):
        super().__init__()
        self.weight = weight
        
        if freq_bands is None:
            freq_bands = [(8, 13), (13, 30)]  # Alpha and Beta bands
        
        self.freq_bands = freq_bands
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute phase consistency loss.
        
        Args:
            inputs: Dictionary with 'predictions' and 'targets'
            
        Returns:
            Phase consistency loss tensor
        """
        predictions = inputs['predictions']['reconstruction']
        targets = inputs['targets']
        
        total_loss = 0.0
        
        for low_freq, high_freq in self.freq_bands:
            # Filter signals to frequency band
            pred_filtered = self._bandpass_filter(predictions, low_freq, high_freq)
            target_filtered = self._bandpass_filter(targets, low_freq, high_freq)
            
            # Extract instantaneous phase
            pred_phase = self._instantaneous_phase(pred_filtered)
            target_phase = self._instantaneous_phase(target_filtered)
            
            # Phase consistency loss (circular distance)
            phase_diff = torch.angle(torch.exp(1j * (pred_phase - target_phase)))
            phase_loss = torch.mean(torch.abs(phase_diff))
            
            total_loss += phase_loss
        
        return total_loss / len(self.freq_bands)
    
    def _bandpass_filter(self, signal: torch.Tensor, low_freq: float, high_freq: float) -> torch.Tensor:
        """
        Simple FFT-based bandpass filter.
        """
        # Compute FFT
        fft = torch.fft.fft(signal, dim=-1)
        
        # Create frequency mask
        n_freq = signal.shape[-1]
        freqs = torch.fft.fftfreq(n_freq, d=1.0/1000.0)  # Assuming 1000 Hz sampling rate
        
        freq_mask = (torch.abs(freqs) >= low_freq) & (torch.abs(freqs) <= high_freq)
        freq_mask = freq_mask.to(signal.device)
        
        # Apply filter
        filtered_fft = fft * freq_mask
        filtered_signal = torch.fft.ifft(filtered_fft, dim=-1).real
        
        return filtered_signal
    
    def _instantaneous_phase(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Compute instantaneous phase using Hilbert transform.
        """
        # Hilbert transform via FFT
        fft = torch.fft.fft(signal, dim=-1)
        n = signal.shape[-1]
        
        # Create Hilbert transform weights
        h = torch.zeros(n, device=signal.device)
        if n % 2 == 0:
            h[0] = h[n//2] = 1
            h[1:n//2] = 2
        else:
            h[0] = 1
            h[1:(n+1)//2] = 2
        
        # Apply Hilbert transform
        analytic_signal = torch.fft.ifft(fft * h, dim=-1)
        
        # Extract phase
        phase = torch.angle(analytic_signal)
        
        return phase


class NeuralComplexityLoss(nn.Module):
    """
    Loss to encourage realistic neural complexity.
    Based on measures like Lempel-Ziv complexity or multiscale entropy.
    """
    
    def __init__(self, weight: float = 1.0, complexity_measure: str = 'sample_entropy'):
        super().__init__()
        self.weight = weight
        self.complexity_measure = complexity_measure
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute neural complexity loss.
        
        Args:
            inputs: Dictionary with 'predictions' and 'targets'
            
        Returns:
            Complexity loss tensor
        """
        predictions = inputs['predictions']['reconstruction']
        targets = inputs['targets']
        
        if self.complexity_measure == 'sample_entropy':
            pred_complexity = self._sample_entropy(predictions)
            target_complexity = self._sample_entropy(targets)
        else:
            raise ValueError(f"Unknown complexity measure: {self.complexity_measure}")
        
        complexity_loss = F.mse_loss(pred_complexity, target_complexity)
        
        return complexity_loss
    
    def _sample_entropy(self, signal: torch.Tensor, m: int = 2, r: float = 0.2) -> torch.Tensor:
        """
        Compute sample entropy as a measure of signal complexity.
        
        Args:
            signal: Input signal (batch, channels, time)
            m: Pattern length
            r: Tolerance for matching
            
        Returns:
            Sample entropy values (batch, channels)
        """
        batch_size, n_channels, n_time = signal.shape
        
        # Simplified sample entropy computation (approximation for efficiency)
        entropies = torch.zeros(batch_size, n_channels, device=signal.device)
        
        for b in range(batch_size):
            for ch in range(n_channels):
                signal_ch = signal[b, ch, :]
                
                # Normalize signal
                signal_ch = (signal_ch - signal_ch.mean()) / (signal_ch.std() + 1e-8)
                
                # Count template matches
                templates_m = []
                templates_m1 = []
                
                for i in range(n_time - m):
                    template_m = signal_ch[i:i+m]
                    template_m1 = signal_ch[i:i+m+1] if i < n_time - m else None
                    
                    templates_m.append(template_m)
                    if template_m1 is not None:
                        templates_m1.append(template_m1)
                
                # Count matches within tolerance r
                if len(templates_m) > 0:
                    templates_m = torch.stack(templates_m)
                    matches_m = 0
                    matches_m1 = 0
                    
                    for i in range(len(templates_m)):
                        distances_m = torch.max(torch.abs(templates_m - templates_m[i]), dim=1)[0]
                        matches_m += torch.sum(distances_m <= r).item()
                        
                        if i < len(templates_m1):
                            templates_m1_tensor = torch.stack(templates_m1)
                            distances_m1 = torch.max(torch.abs(templates_m1_tensor - templates_m1[i]), dim=1)[0]
                            matches_m1 += torch.sum(distances_m1 <= r).item()
                    
                    # Compute sample entropy
                    if matches_m > 0 and matches_m1 > 0:
                        sample_entropy = -torch.log(torch.tensor(matches_m1 / matches_m, dtype=torch.float32))
                        entropies[b, ch] = sample_entropy
        
        return entropies