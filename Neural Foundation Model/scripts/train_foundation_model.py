#!/usr/bin/env python3
"""
Neural Foundation Model Training Script

Distributed training pipeline for neural foundation models with support for
temporal parallelism, privacy-preserving training, and real-time monitoring.
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from omegaconf import DictConfig, OmegaConf
import wandb
import mlflow
from accelerate import Accelerator
import ray
from ray import tune
from ray.air import session as ray_session

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_foundation.models.foundation_model import NeuralFoundationModel
from neural_foundation.data.streaming_loader import NeuralDataLoader, StreamingNeuralDataset
from neural_foundation.training.losses import (
    MultiTaskLoss, ContrastiveLoss, ReconstructionLoss, 
    TemporalConsistencyLoss, SubjectInvarianceLoss
)
from neural_foundation.training.optimizers import get_optimizer, get_scheduler
from neural_foundation.training.callbacks import (
    ModelCheckpointer, EarlyStopping, PerformanceMonitor
)
from neural_foundation.privacy.differential_privacy import DPTrainer
from neural_foundation.monitoring.metrics import NeuralTrainingMetrics


class NeuralFoundationTrainer:
    """
    Main trainer class for neural foundation models.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            mixed_precision=config.training.mixed_precision,
            log_with=config.logging.backend if config.logging.enabled else None
        )
        
        self.is_main_process = self.accelerator.is_main_process
        self.device = self.accelerator.device
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Monitoring
        self.metrics = NeuralTrainingMetrics()
        
        # Callbacks
        self.callbacks = []
        
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.logging.level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if self.is_main_process and self.config.logging.enabled:
            if self.config.logging.backend == "wandb":
                wandb.init(
                    project=self.config.logging.wandb.project,
                    name=self.config.logging.wandb.run_name,
                    config=OmegaConf.to_container(self.config, resolve=True)
                )
            elif self.config.logging.backend == "mlflow":
                mlflow.set_experiment(self.config.logging.mlflow.experiment_name)
                mlflow.start_run(run_name=self.config.logging.mlflow.run_name)
    
    def initialize_model(self):
        """Initialize the neural foundation model."""
        self.logger.info("Initializing neural foundation model...")
        
        model_config = {
            'num_subjects': self.config.data.num_subjects,
            'enable_privacy': self.config.privacy.enabled,
            'noise_multiplier': self.config.privacy.noise_multiplier
        }
        
        self.model = NeuralFoundationModel(
            config=model_config,
            num_channels=self.config.model.num_channels,
            sampling_rate=self.config.data.sampling_rate,
            context_length=self.config.model.context_length,
            hidden_dim=self.config.model.hidden_dim,
            num_layers=self.config.model.num_layers,
            num_heads=self.config.model.num_heads,
            dropout=self.config.model.dropout
        )
        
        # Load from checkpoint if specified
        if self.config.training.resume_from_checkpoint:
            self.load_checkpoint(self.config.training.resume_from_checkpoint)
        
        self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def initialize_data_loaders(self):
        """Initialize data loaders."""
        self.logger.info("Initializing data loaders...")
        
        # Training data loader
        self.train_loader = NeuralDataLoader(
            data_path=self.config.data.train_path,
            batch_size=self.config.training.batch_size,
            temporal_window=self.config.data.temporal_window,
            sampling_rate=self.config.data.sampling_rate,
            channels=self.config.data.channels,
            preprocessing_config=self.config.data.preprocessing,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
        
        # Validation data loader
        if self.config.data.val_path:
            self.val_loader = NeuralDataLoader(
                data_path=self.config.data.val_path,
                batch_size=self.config.training.val_batch_size,
                temporal_window=self.config.data.temporal_window,
                sampling_rate=self.config.data.sampling_rate,
                channels=self.config.data.channels,
                preprocessing_config=self.config.data.preprocessing,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                pin_memory=True
            )
        
        self.logger.info("Data loaders initialized")
    
    def initialize_loss_function(self):
        """Initialize the loss function."""
        loss_components = {}
        
        # Reconstruction loss
        if self.config.loss.reconstruction.enabled:
            loss_components['reconstruction'] = ReconstructionLoss(
                loss_type=self.config.loss.reconstruction.type,
                weight=self.config.loss.reconstruction.weight
            )
        
        # Contrastive loss
        if self.config.loss.contrastive.enabled:
            loss_components['contrastive'] = ContrastiveLoss(
                temperature=self.config.loss.contrastive.temperature,
                weight=self.config.loss.contrastive.weight
            )
        
        # Temporal consistency loss
        if self.config.loss.temporal_consistency.enabled:
            loss_components['temporal_consistency'] = TemporalConsistencyLoss(
                weight=self.config.loss.temporal_consistency.weight
            )
        
        # Subject invariance loss
        if self.config.loss.subject_invariance.enabled:
            loss_components['subject_invariance'] = SubjectInvarianceLoss(
                weight=self.config.loss.subject_invariance.weight
            )
        
        self.loss_fn = MultiTaskLoss(loss_components)
        self.logger.info(f"Loss function initialized with components: {list(loss_components.keys())}")
    
    def initialize_optimizer_and_scheduler(self):
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = get_optimizer(
            model=self.model,
            optimizer_config=self.config.optimizer
        )
        
        if self.config.scheduler.enabled:
            self.scheduler = get_scheduler(
                optimizer=self.optimizer,
                scheduler_config=self.config.scheduler
            )
        
        self.logger.info("Optimizer and scheduler initialized")
    
    def initialize_callbacks(self):
        """Initialize training callbacks."""
        # Model checkpointer
        if self.config.callbacks.checkpointing.enabled:
            self.callbacks.append(
                ModelCheckpointer(
                    checkpoint_dir=self.config.callbacks.checkpointing.save_dir,
                    save_every_n_epochs=self.config.callbacks.checkpointing.save_every_n_epochs,
                    save_top_k=self.config.callbacks.checkpointing.save_top_k
                )
            )
        
        # Early stopping
        if self.config.callbacks.early_stopping.enabled:
            self.callbacks.append(
                EarlyStopping(
                    patience=self.config.callbacks.early_stopping.patience,
                    min_delta=self.config.callbacks.early_stopping.min_delta
                )
            )
        
        # Performance monitoring
        self.callbacks.append(
            PerformanceMonitor(
                log_every_n_steps=self.config.logging.log_every_n_steps
            )
        )
        
        self.logger.info(f"Initialized {len(self.callbacks)} callbacks")
    
    def setup_distributed_training(self):
        """Setup distributed training components."""
        # Prepare model and data loaders with accelerator
        if self.train_loader is not None:
            self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader
            )
        
        if self.val_loader is not None:
            self.val_loader = self.accelerator.prepare(self.val_loader)
        
        if self.scheduler is not None:
            self.scheduler = self.accelerator.prepare(self.scheduler)
        
        self.logger.info("Distributed training setup complete")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {}
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            with self.accelerator.accumulate(self.model):
                # Forward pass
                neural_data = batch['neural_data']
                subject_ids = self._encode_subject_ids(batch['subject_ids'])
                
                outputs = self.model(
                    neural_data=neural_data,
                    subject_ids=subject_ids,
                    output_heads=['reconstruction', 'contrastive']
                )
                
                # Compute loss
                loss_inputs = {
                    'predictions': outputs,
                    'targets': neural_data,
                    'subject_ids': subject_ids
                }
                
                loss_dict = self.loss_fn(loss_inputs)
                total_loss_value = loss_dict['total_loss']
                
                # Backward pass
                self.accelerator.backward(total_loss_value)
                
                # Gradient clipping
                if self.config.training.gradient_clipping.enabled:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clipping.max_norm
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += total_loss_value.item()
                num_batches += 1
                self.global_step += 1
                
                # Log step metrics
                if self.is_main_process and batch_idx % self.config.logging.log_every_n_steps == 0:
                    step_metrics = {
                        'train/step_loss': total_loss_value.item(),
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'train/global_step': self.global_step
                    }
                    
                    # Add individual loss components
                    for loss_name, loss_value in loss_dict.items():
                        if loss_name != 'total_loss':
                            step_metrics[f'train/{loss_name}_loss'] = loss_value.item()
                    
                    self.log_metrics(step_metrics)
                
                # Apply callbacks
                for callback in self.callbacks:
                    if hasattr(callback, 'on_step_end'):
                        callback.on_step_end(self, batch_idx, loss_dict)
        
        # Compute epoch metrics
        epoch_metrics['train/epoch_loss'] = total_loss / num_batches
        
        # Update learning rate scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_metrics = {}
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                neural_data = batch['neural_data']
                subject_ids = self._encode_subject_ids(batch['subject_ids'])
                
                outputs = self.model(
                    neural_data=neural_data,
                    subject_ids=subject_ids,
                    output_heads=['reconstruction', 'contrastive']
                )
                
                # Compute loss
                loss_inputs = {
                    'predictions': outputs,
                    'targets': neural_data,
                    'subject_ids': subject_ids
                }
                
                loss_dict = self.loss_fn(loss_inputs)
                total_loss += loss_dict['total_loss'].item()
                num_batches += 1
        
        val_metrics['val/epoch_loss'] = total_loss / num_batches
        return val_metrics
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['epoch'] = epoch
            epoch_metrics['epoch_time'] = time.time() - epoch_start_time
            
            # Log epoch metrics
            if self.is_main_process:
                self.log_metrics(epoch_metrics)
                self.logger.info(
                    f"Epoch {epoch}: train_loss={train_metrics.get('train/epoch_loss', 'N/A'):.4f}, "
                    f"val_loss={val_metrics.get('val/epoch_loss', 'N/A'):.4f}, "
                    f"time={epoch_metrics['epoch_time']:.2f}s"
                )
            
            # Apply callbacks
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(self, epoch, epoch_metrics)
            
            # Check early stopping
            val_loss = val_metrics.get('val/epoch_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                
                # Save best model
                if self.is_main_process:
                    self.save_checkpoint(
                        os.path.join(self.config.training.output_dir, 'best_model.pt'),
                        is_best=True
                    )
        
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.accelerator.get_state_dict(self.model),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            self.logger.info(f"Saved best model checkpoint to {filepath}")
        else:
            self.logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from {filepath} (epoch {self.current_epoch})")
    
    def _encode_subject_ids(self, subject_ids: List[str]) -> torch.Tensor:
        """Encode subject IDs to integers."""
        # Simple hash-based encoding (in practice, use a proper mapping)
        encoded = []
        for subject_id in subject_ids:
            encoded.append(hash(subject_id) % self.config.data.num_subjects)
        return torch.tensor(encoded, device=self.device)
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to tracking backend."""
        if not self.config.logging.enabled:
            return
        
        if self.config.logging.backend == "wandb":
            wandb.log(metrics, step=self.global_step)
        elif self.config.logging.backend == "mlflow":
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=self.global_step)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.config.logging.enabled and self.is_main_process:
            if self.config.logging.backend == "wandb":
                wandb.finish()
            elif self.config.logging.backend == "mlflow":
                mlflow.end_run()


def train_with_ray_tune(config: Dict) -> Dict[str, float]:
    """Training function for Ray Tune hyperparameter optimization."""
    # Convert dict config to DictConfig
    cfg = OmegaConf.create(config)
    
    # Initialize trainer
    trainer = NeuralFoundationTrainer(cfg)
    
    try:
        # Initialize all components
        trainer.initialize_model()
        trainer.initialize_data_loaders()
        trainer.initialize_loss_function()
        trainer.initialize_optimizer_and_scheduler()
        trainer.initialize_callbacks()
        trainer.setup_distributed_training()
        
        # Training loop with Ray Tune reporting
        for epoch in range(cfg.training.num_epochs):
            trainer.current_epoch = epoch
            
            # Train and validate
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.validate_epoch()
            
            # Report to Ray Tune
            metrics_to_report = {
                "epoch": epoch,
                "train_loss": train_metrics.get('train/epoch_loss', 0.0),
                "val_loss": val_metrics.get('val/epoch_loss', 0.0)
            }
            
            ray_session.report(metrics_to_report)
        
        # Return final validation loss
        return {"val_loss": val_metrics.get('val/epoch_loss', float('inf'))}
    
    finally:
        trainer.cleanup()


@hydra.main(version_base=None, config_path="../configs/training", config_name="foundation_model")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Create output directory
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    
    # Hyperparameter optimization mode
    if cfg.training.hyperparameter_optimization.enabled:
        run_hyperparameter_optimization(cfg)
    else:
        # Standard training mode
        trainer = NeuralFoundationTrainer(cfg)
        
        try:
            # Initialize all components
            trainer.initialize_model()
            trainer.initialize_data_loaders()
            trainer.initialize_loss_function()
            trainer.initialize_optimizer_and_scheduler()
            trainer.initialize_callbacks()
            trainer.setup_distributed_training()
            
            # Train the model
            trainer.train()
            
            # Save final model
            if trainer.is_main_process:
                final_model_path = os.path.join(cfg.training.output_dir, 'final_model.pt')
                trainer.model.save_pretrained(final_model_path)
                trainer.logger.info(f"Final model saved to {final_model_path}")
        
        finally:
            trainer.cleanup()


def run_hyperparameter_optimization(cfg: DictConfig):
    """Run hyperparameter optimization with Ray Tune."""
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Define search space
    search_space = {
        "model": {
            "hidden_dim": tune.choice([256, 512, 768, 1024]),
            "num_layers": tune.choice([6, 8, 12, 16]),
            "num_heads": tune.choice([4, 8, 12, 16]),
            "dropout": tune.uniform(0.0, 0.3)
        },
        "optimizer": {
            "lr": tune.loguniform(1e-5, 1e-3),
            "weight_decay": tune.loguniform(1e-6, 1e-2)
        },
        "training": {
            "batch_size": tune.choice([4, 8, 16, 32]),
            "gradient_accumulation_steps": tune.choice([1, 2, 4, 8])
        }
    }
    
    # Merge with base config
    base_config = OmegaConf.to_container(cfg, resolve=True)
    
    # ASHA scheduler for early termination
    scheduler = ASHAScheduler(
        time_attr='epoch',
        metric='val_loss',
        mode='min',
        max_t=cfg.training.num_epochs,
        grace_period=10,
        reduction_factor=3
    )
    
    # Run tuning
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_with_ray_tune),
            resources={"cpu": 4, "gpu": 1}
        ),
        param_space={**base_config, **search_space},
        tune_config=tune.TuneConfig(
            num_samples=cfg.training.hyperparameter_optimization.num_trials,
            scheduler=scheduler,
            metric="val_loss",
            mode="min"
        ),
        run_config=ray.air.RunConfig(
            name="neural_foundation_hpo",
            local_dir=cfg.training.output_dir
        )
    )
    
    results = tuner.fit()
    
    # Get best trial
    best_trial = results.get_best_result(metric="val_loss", mode="min")
    print(f"Best trial: {best_trial.config}")
    print(f"Best validation loss: {best_trial.metrics['val_loss']}")


def setup_differential_privacy_training(cfg: DictConfig, model: nn.Module, optimizer) -> DPTrainer:
    """Setup differential privacy training."""
    dp_trainer = DPTrainer(
        model=model,
        optimizer=optimizer,
        noise_multiplier=cfg.privacy.noise_multiplier,
        max_grad_norm=cfg.privacy.max_grad_norm,
        delta=cfg.privacy.delta,
        epochs=cfg.training.num_epochs,
        batch_size=cfg.training.batch_size,
        sample_size=cfg.privacy.sample_size
    )
    
    return dp_trainer


if __name__ == "__main__":
    # Handle command line arguments
    parser = argparse.ArgumentParser(description="Train Neural Foundation Model")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--data_path", type=str, help="Path to training data")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--ray", action="store_true", help="Use Ray for distributed training")
    
    args = parser.parse_args()
    
    # Set environment variables for distributed training
    if args.distributed:
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"
    
    # Override config with command line arguments
    overrides = []
    if args.data_path:
        overrides.append(f"data.train_path={args.data_path}")
    if args.output_dir:
        overrides.append(f"training.output_dir={args.output_dir}")
    if args.resume:
        overrides.append(f"training.resume_from_checkpoint={args.resume}")
    
    # Update sys.argv for Hydra
    if overrides:
        sys.argv.extend(overrides)
    
    # Run main function
    main()