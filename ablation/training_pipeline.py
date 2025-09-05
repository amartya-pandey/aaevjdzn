"""
Training pipeline for the dual-path kidney classification model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import time
from pathlib import Path

from config import TRAINING_CONFIG, CLASS_NAMES, DATASET_STATS
from models import create_model
from loss import create_loss_function
from utils import compute_metrics, thresholded_predictions


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class KidneyTrainer:
    """Training pipeline for kidney condition classification."""
    
    def __init__(self, model, train_loader, val_loader, device='cpu', 
                 loss_type='combined', save_dir='./checkpoints', use_freezing=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.use_freezing = use_freezing
        
        # Move model to device
        self.model.to(device)
        
        # Apply freezing if enabled
        if use_freezing and hasattr(self.model, 'freeze_for_dual_path'):
            print("🧊 Applying dual-path optimized freezing...")
            self.model.freeze_for_dual_path()
            self.model.print_freezing_summary()
            
            # Use parameter groups with different learning rates
            param_groups = self.model.get_parameter_groups(TRAINING_CONFIG['learning_rate'])
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=TRAINING_CONFIG['weight_decay']
            )
        else:
            # Standard optimizer for all parameters
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=TRAINING_CONFIG['learning_rate'],
                weight_decay=TRAINING_CONFIG['weight_decay']
            )
        
        # Loss function
        self.criterion = create_loss_function(loss_type)
        self.criterion.to(device)  # Move loss function to device
        
        # Learning rate scheduler
        if hasattr(TRAINING_CONFIG, 'scheduler_step_size'):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=TRAINING_CONFIG['scheduler_step_size'],
                gamma=TRAINING_CONFIG['scheduler_gamma']
            )
        else:
            # Use ReduceLROnPlateau for better performance with freezing
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=TRAINING_CONFIG['patience'],
            restore_best_weights=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rate': []
        }
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, confidence = self.model(images)
            
            # Compute loss
            loss, loss_dict = self.criterion(logits, confidence, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            probabilities = torch.softmax(logits, dim=1)
            predicted = thresholded_predictions(probabilities)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100 * correct_predictions / total_samples:.2f}%')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Forward pass
                logits, confidence = self.model(images)
                
                # Compute loss
                loss, _ = self.criterion(logits, confidence, targets)
                total_loss += loss.item()
                
                # Predictions
                probabilities = torch.softmax(logits, dim=1)
                predicted = thresholded_predictions(probabilities)
                
                # Store for metrics
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Compute metrics
        metrics = compute_metrics(
            np.array(all_targets), 
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        return avg_loss, metrics
    
    def train(self, num_epochs=None):
        """Full training loop."""
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG['num_epochs']
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Gradual unfreezing (if enabled and model supports it)
            if (self.use_freezing and hasattr(self.model, 'unfreeze_gradually')):
                self.model.unfreeze_gradually(epoch, num_epochs)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_metrics = self.validate_epoch()
            val_acc = val_metrics['accuracy']
            val_f1 = val_metrics['macro_f1']
            
            # Learning rate step (handle different scheduler types)
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f'\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_f1)
            
            # Early stopping
            if self.early_stopping(val_f1, self.model):
                print(f'\nEarly stopping triggered at epoch {epoch+1}')
                break
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time:.1f} seconds')
        
        return self.history
    
    def save_checkpoint(self, epoch, val_f1):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_f1': val_f1,
            'history': self.history
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pth')
        
        # Save best checkpoint if this is the best validation F1
        if not hasattr(self, 'best_val_f1') or val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            torch.save(checkpoint, self.save_dir / 'best_checkpoint.pth')
            print(f'Saved best checkpoint with val F1: {val_f1:.4f}')
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
              f"with val F1: {checkpoint['val_f1']:.4f}")
        
        return checkpoint['epoch']


def create_weighted_sampler(targets):
    """
    Create weighted sampler for balanced training.
    
    Args:
        targets: Training targets
        
    Returns:
        sampler: WeightedRandomSampler instance
    """
    # Count samples per class
    class_counts = np.bincount(targets)
    
    # Calculate weights (inverse frequency)
    class_weights = 1.0 / class_counts
    
    # Assign weight to each sample
    sample_weights = class_weights[targets]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def create_trainer(train_dataset, val_dataset, device='cpu', **kwargs):
    """
    Factory function to create trainer with dataloaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device: Device for training
        **kwargs: Additional arguments for trainer
        
    Returns:
        trainer: KidneyTrainer instance
    """
    # Create weighted sampler for training
    train_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
    train_sampler = create_weighted_sampler(train_targets)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True if device != 'cpu' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if device != 'cpu' else False
    )
    
    # Create model
    model = create_model()
    
    # Create trainer
    trainer = KidneyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        **kwargs
    )
    
    return trainer


if __name__ == "__main__":
    print("Training pipeline module loaded successfully!")
    print(f"Training configuration: {TRAINING_CONFIG}")
    print(f"Available classes: {CLASS_NAMES}")
    print(f"Dataset stats: {DATASET_STATS}")
