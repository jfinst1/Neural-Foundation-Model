#!/usr/bin/env python3
"""
Real-time Neural Decoding Inference Server

High-performance streaming inference server for neural foundation models.
Optimized for <1ms latency brain-computer interface applications.
"""

import asyncio
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import signal
import sys

import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_foundation.models.foundation_model import NeuralFoundationModel, TaskSpecificDecoder
from neural_foundation.signal_processing.filters import CausalButterworthFilter, CUDASignalProcessor
from neural_foundation.signal_processing.artifacts import RealTimeArtifactProcessor
from neural_foundation.data.streaming_loader import MockAcquisitionInterface


class NeuralSample(BaseModel):
    """Input neural data sample."""
    data: List[List[float]]  # (channels, time_samples)
    timestamp: float
    sampling_rate: int = 1000
    subject_id: str = "default"


class DecodingResult(BaseModel):
    """Neural decoding result."""
    prediction: List[float]
    confidence: float
    latency_ms: float
    timestamp: float
    artifact_detected: bool


class RealTimeNeuralDecoder:
    """
    Real-time neural decoder with <1ms latency optimization.
    """
    
    def __init__(
        self,
        model_path: str,
        task: str = "motor_control",
        device: str = "cuda",
        max_batch_size: int = 1,
        optimization_level: str = "aggressive"
    ):
        self.model_path = model_path
        self.task = task
        self.device = torch.device(device)
        self.max_batch_size = max_batch_size
        self.optimization_level = optimization_level
        
        # Performance monitoring
        self.inference_times = []
        self.total_samples_processed = 0
        
        # Initialize components
        self._load_model()
        self._setup_preprocessing()
        self._optimize_model()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Real-time decoder initialized for task: {task}")
    
    def _load_model(self):
        """Load and setup the neural foundation model."""
        self.logger.info(f"Loading model from {self.model_path}")
        
        # Load foundation model
        self.foundation_model = NeuralFoundationModel.from_pretrained(self.model_path)
        self.foundation_model.to(self.device)
        self.foundation_model.eval()
        
        # Create task-specific decoder
        self.decoder = TaskSpecificDecoder(self.foundation_model, self.task)
        self.decoder.to(self.device)
        self.decoder.eval()
        
        # Disable gradients for inference
        for param in self.foundation_model.parameters():
            param.requires_grad_(False)
        for param in self.decoder.parameters():
            param.requires_grad_(False)
    
    def _setup_preprocessing(self):
        """Setup signal preprocessing pipeline."""
        # Real-time signal processor
        self.signal_processor = CUDASignalProcessor(
            num_channels=64,  # TODO: Get from model config
            sampling_rate=1000.0,
            filter_configs={
                'bandpass': {'low_freq': 1.0, 'high_freq': 100.0, 'order': 4},
                'notch': {'freq': 50.0, 'quality': 30.0}
            }
        ).to(self.device)
        
        # Artifact detector
        self.artifact_processor = RealTimeArtifactProcessor(
            sampling_rate=1000.0,
            buffer_size=2.0,
            num_channels=64,
            detection_methods=['amplitude', 'gradient']
        )
        
        # Context buffer for temporal models
        self.context_length = 1000  # 1 second at 1kHz
        self.context_buffer = torch.zeros(
            1, 64, self.context_length, device=self.device, dtype=torch.float32
        )
        
    def _optimize_model(self):
        """Apply aggressive optimizations for minimal latency."""
        if self.optimization_level == "aggressive":
            # Compile model with TorchScript
            self.logger.info("Compiling model with TorchScript...")
            
            # Create example input for tracing
            example_input = torch.randn(1, 64, self.context_length, device=self.device)
            
            try:
                # Trace the decoder
                self.decoder = torch.jit.trace(self.decoder, example_input)
                self.decoder = torch.jit.optimize_for_inference(self.decoder)
                self.logger.info("TorchScript compilation successful")
            except Exception as e:
                self.logger.warning(f"TorchScript compilation failed: {e}")
            
            # Enable CUDA optimizations
            if self.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Use mixed precision for faster inference
                self.use_amp = True
                self.autocast = torch.cuda.amp.autocast()
            else:
                self.use_amp = False
    
    async def decode_sample(self, neural_sample: NeuralSample) -> DecodingResult:
        """
        Decode a single neural sample with minimal latency.
        
        Args:
            neural_sample: Input neural data sample
            
        Returns:
            Decoding result with prediction and metadata
        """
        start_time = time.perf_counter()
        
        # Convert input to tensor
        data = torch.tensor(neural_sample.data, device=self.device, dtype=torch.float32)
        
        if data.dim() == 2:
            data = data.unsqueeze(0)  # Add batch dimension
        
        # Real-time preprocessing
        processed_data, artifact_detected = await self._preprocess_sample(data)
        
        # Update context buffer
        self._update_context_buffer(processed_data)
        
        # Neural decoding
        if self.use_amp:
            with torch.cuda.amp.autocast():
                prediction = await self._decode_context()
        else:
            prediction = await self._decode_context()
        
        # Compute confidence score
        confidence = self._compute_confidence(prediction)
        
        # Performance tracking
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        self.inference_times.append(latency_ms)
        self.total_samples_processed += 1
        
        # Keep only recent timing data
        if len(self.inference_times) > 1000:
            self.inference_times.pop(0)
        
        return DecodingResult(
            prediction=prediction.cpu().numpy().tolist(),
            confidence=confidence,
            latency_ms=latency_ms,
            timestamp=neural_sample.timestamp,
            artifact_detected=artifact_detected
        )
    
    async def _preprocess_sample(self, data: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Preprocess neural sample with artifact detection.
        
        Args:
            data: Raw neural data tensor
            
        Returns:
            Tuple of (processed_data, artifact_detected)
        """
        # Apply signal processing filters
        filtered_data = self.signal_processor.filter(data)
        
        # Artifact detection and cleaning
        data_np = data.squeeze(0).cpu().numpy()
        processed_sample, artifact_detected = self.artifact_processor.process_sample(data_np)
        
        # Convert back to tensor
        processed_data = torch.from_numpy(processed_sample).unsqueeze(0).to(self.device)
        
        return processed_data, artifact_detected
    
    def _update_context_buffer(self, data: torch.Tensor):
        """Update the temporal context buffer with new data."""
        batch_size, channels, time_samples = data.shape
        
        if time_samples >= self.context_length:
            # Replace entire buffer
            self.context_buffer = data[:, :, -self.context_length:].contiguous()
        else:
            # Shift buffer and add new samples
            self.context_buffer[:, :, :-time_samples] = self.context_buffer[:, :, time_samples:].clone()
            self.context_buffer[:, :, -time_samples:] = data
    
    async def _decode_context(self) -> torch.Tensor:
        """
        Decode current context buffer.
        
        Returns:
            Decoded prediction tensor
        """
        # Run inference on current context
        with torch.no_grad():
            prediction = self.decoder(self.context_buffer)
        
        return prediction
    
    def _compute_confidence(self, prediction: torch.Tensor) -> float:
        """
        Compute confidence score for the prediction.
        
        Args:
            prediction: Model prediction tensor
            
        Returns:
            Confidence score between 0 and 1
        """
        if self.task == "motor_control":
            # For continuous outputs, use inverse of prediction uncertainty
            prediction_std = torch.std(prediction).item()
            confidence = 1.0 / (1.0 + prediction_std)
        
        elif self.task in ["motor_imagery", "seizure_detection"]:
            # For classification, use softmax probability
            probs = torch.softmax(prediction, dim=-1)
            confidence = torch.max(probs).item()
        
        else:
            # Default confidence based on L2 norm
            confidence = 1.0 / (1.0 + torch.norm(prediction).item())
        
        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.inference_times:
            return {"status": "No samples processed yet"}
        
        inference_times = np.array(self.inference_times)
        
        return {
            "total_samples": self.total_samples_processed,
            "mean_latency_ms": float(np.mean(inference_times)),
            "median_latency_ms": float(np.median(inference_times)),
            "p95_latency_ms": float(np.percentile(inference_times, 95)),
            "p99_latency_ms": float(np.percentile(inference_times, 99)),
            "max_latency_ms": float(np.max(inference_times)),
            "throughput_hz": 1000.0 / np.mean(inference_times) if len(inference_times) > 0 else 0.0
        }


# Global decoder instance
decoder: Optional[RealTimeNeuralDecoder] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager."""
    global decoder
    
    # Startup
    model_path = getattr(app.state, 'model_path', './models/foundation_model.pt')
    task = getattr(app.state, 'task', 'motor_control')
    device = getattr(app.state, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    decoder = RealTimeNeuralDecoder(
        model_path=model_path,
        task=task,
        device=device
    )
    
    yield
    
    # Shutdown
    decoder = None


# FastAPI application
app = FastAPI(
    title="Neural Real-time Decoder",
    description="High-performance streaming inference for neural foundation models",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/decode", response_model=DecodingResult)
async def decode_neural_sample(sample: NeuralSample):
    """Decode a single neural sample."""
    if decoder is None:
        raise HTTPException(status_code=500, detail="Decoder not initialized")
    
    try:
        result = await decoder.decode_sample(sample)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decoding failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if decoder is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Decoder not initialized"}
        )
    
    return {"status": "healthy", "device": str(decoder.device)}


@app.get("/stats")
async def get_stats():
    """Get performance statistics."""
    if decoder is None:
        raise HTTPException(status_code=500, detail="Decoder not initialized")
    
    return decoder.get_performance_stats()


@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming neural decoding."""
    await websocket.accept()
    
    if decoder is None:
        await websocket.close(code=1011, reason="Decoder not initialized")
        return
    
    try:
        while True:
            # Receive neural data
            data = await websocket.receive_text()
            sample_data = json.loads(data)
            
            # Create neural sample
            neural_sample = NeuralSample(**sample_data)
            
            # Decode sample
            result = await decoder.decode_sample(neural_sample)
            
            # Send result back
            await websocket.send_text(result.json())
            
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


class StreamingClient:
    """
    Client for testing streaming neural decoding.
    """
    
    def __init__(self, server_url: str = "ws://localhost:8080/stream"):
        self.server_url = server_url
        self.mock_acquisition = MockAcquisitionInterface(
            num_channels=64,
            sampling_rate=1000
        )
        
    async def start_streaming(self, duration: float = 10.0):
        """Start streaming test data to the server."""
        import websockets
        
        try:
            async with websockets.connect(self.