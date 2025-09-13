"""
Privacy module for neural foundation models.

Provides differential privacy, data anonymization, and federated learning
capabilities for protecting sensitive brain data.
"""

from .differential_privacy import (
    DifferentiallyPrivateLayer,
    DPTrainer,
    NeuralDataDPSampler,
    PrivacyAuditLogger
)

from .anonymization import (
    NeuralDataAnonymizer,
    BaseAnonymizer,
    DifferentialPrivacyAnonymizer,
    KAnonymityProcessor,
    NoiseInjectionAnonymizer
)

__all__ = [
    # Differential Privacy
    'DifferentiallyPrivateLayer',
    'DPTrainer', 
    'NeuralDataDPSampler',
    'PrivacyAuditLogger',
    
    # Anonymization
    'NeuralDataAnonymizer',
    'BaseAnonymizer',
    'DifferentialPrivacyAnonymizer',
    'KAnonymityProcessor',
    'NoiseInjectionAnonymizer',
]