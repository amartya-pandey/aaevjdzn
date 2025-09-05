#!/usr/bin/env python3
"""
Ablation Study Script for Dual-Path Kidney Classification System

This script runs ablation studies to evaluate the contribution of each component:
1. EfficientNet-B4 only (Path A)
2. Lightweight CNN only (Path B) 
3. Dual-path with simple fusion
4. Full dual-path with cross-attention (baseline)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from config import ABLATION_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, CLASS_NAMES
from models import create_model
from training_pipeline import KidneyTrainer
from utils import compute_metrics, thresholded_predictions
from dataloader import create_datasets, create_dataloaders


class AblationStudy:
    """Class to manage ablation study experiments."""
    
    def __init__(self, data_dir, device='cuda', save_dir='./ablation_results'):
        self.data_dir = data_dir
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Load datasets once for all experiments
        print("Loading datasets...")
        self.datasets = create_datasets(data_dir)
        self.dataloaders = create_dataloaders(
            self.datasets, 
            batch_size=TRAINING_CONFIG['batch_size']
        )
        
        # Storage for results
        self.results = {}
        
    def run_single_experiment(self, model_type, experiment_name):
        """Run a single ablation experiment."""
        print(f"\n{'='*60}")
        print(f"Running: {experiment_name}")
        print(f"Model type: {model_type}")
        print(f"{'='*60}")
        
        # Create model
        model = create_model(model_type)
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Create trainer (disable freezing for ablation studies)
        trainer = KidneyTrainer(
            model=model,
            train_loader=self.dataloaders['train'],
            val_loader=self.dataloaders['val'],
            device=self.device,
            loss_type='combined',
            use_freezing=False,  # Disable freezing for fair comparison
            save_dir=self.save_dir / model_type
        )
        
        # Training
        start_time = time.time()
        history = trainer.train(num_epochs=TRAINING_CONFIG['num_epochs'])
        training_time = time.time() - start_time
        
        # Evaluation on test set
        test_metrics = self.evaluate_model(trainer.model, self.dataloaders['test'])
        
        # Inference time measurement
        inference_time = self.measure_inference_time(trainer.model)
        
        # Store results
        self.results[model_type] = {
            'experiment_name': experiment_name,
            'model_type': model_type,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'training_time': training_time,
            'inference_time': inference_time,
            'history': history,
            'test_metrics': test_metrics,
            'best_val_f1': max(history['val_f1']),
            'final_val_acc': history['val_acc'][-1],
            'final_train_acc': history['train_acc'][-1]
        }
        
        # Save individual results
        self.save_experiment_results(model_type)
        
        print(f"\n✅ Completed: {experiment_name}")
        print(f"Best Val F1: {max(history['val_f1']):.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Training time: {training_time:.1f}s")
        
        return self.results[model_type]
    
    def evaluate_model(self, model, test_loader):
        """Evaluate model on test set."""
        model.eval()
        all_targets = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                logits, confidence = model(images)
                probabilities = torch.softmax(logits, dim=1)
                predicted = thresholded_predictions(probabilities)
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Compute comprehensive metrics
        metrics = compute_metrics(
            np.array(all_targets),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        return metrics
    
    def measure_inference_time(self, model, num_samples=100):
        """Measure average inference time."""
        model.eval()
        input_size = MODEL_CONFIG['input_size']
        dummy_input = torch.randn(1, 3, input_size, input_size).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Measure
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_samples):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / num_samples * 1000  # ms
        return avg_inference_time
    
    def run_all_experiments(self):
        """Run all ablation study experiments."""
        print("🧪 Starting Ablation Study")
        print(f"Device: {self.device}")
        print(f"Dataset sizes: Train={len(self.datasets['train'])}, "
              f"Val={len(self.datasets['val'])}, Test={len(self.datasets['test'])}")
        
        # Run each experiment
        for variant_key, variant_config in ABLATION_CONFIG['model_variants'].items():
            self.run_single_experiment(
                model_type=variant_config['model_type'],
                experiment_name=variant_config['name']
            )
        
        # Generate comparison report
        self.generate_comparison_report()
        self.plot_results()
        
        print(f"\n🎉 Ablation study completed!")
        print(f"Results saved to: {self.save_dir}")
    
    def save_experiment_results(self, model_type):
        """Save individual experiment results."""
        results = self.results[model_type]
        
        # Save detailed results as JSON
        import json
        results_file = self.save_dir / f"{model_type}_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            k: v.tolist() if isinstance(v, np.ndarray) else v 
            for k, v in results.items() 
            if k != 'history'  # Skip history for now
        }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        print(f"\n📊 Ablation Study Results Summary")
        print(f"{'='*80}")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_type, results in self.results.items():
            comparison_data.append({
                'Model': ABLATION_CONFIG['model_variants'][model_type]['name'],
                'Parameters (M)': results['total_params'] / 1e6,
                'Best Val F1': results['best_val_f1'],
                'Test Accuracy': results['test_metrics']['accuracy'],
                'Test F1': results['test_metrics']['macro_f1'],
                'Training Time (min)': results['training_time'] / 60,
                'Inference Time (ms)': results['inference_time']
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False, float_format='%.3f'))
        
        # Save comparison table
        df.to_csv(self.save_dir / 'ablation_comparison.csv', index=False)
        
        # Find best performing model
        best_model = df.loc[df['Test F1'].idxmax()]
        print(f"\n🏆 Best performing model: {best_model['Model']}")
        print(f"   Test F1: {best_model['Test F1']:.3f}")
        print(f"   Parameters: {best_model['Parameters (M)']:.1f}M")
        
        return df
    
    def plot_results(self):
        """Create visualization plots for ablation study."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Performance comparison
        models = [ABLATION_CONFIG['model_variants'][k]['name'] for k in self.results.keys()]
        test_f1 = [self.results[k]['test_metrics']['macro_f1'] for k in self.results.keys()]
        test_acc = [self.results[k]['test_metrics']['accuracy'] for k in self.results.keys()]
        
        x = np.arange(len(models))
        axes[0, 0].bar(x - 0.2, test_f1, 0.4, label='Test F1', alpha=0.8)
        axes[0, 0].bar(x + 0.2, test_acc, 0.4, label='Test Accuracy', alpha=0.8)
        axes[0, 0].set_xlabel('Model Variants')
        axes[0, 0].set_ylabel('Performance')
        axes[0, 0].set_title('Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Parameter count vs performance
        params = [self.results[k]['total_params'] / 1e6 for k in self.results.keys()]
        axes[0, 1].scatter(params, test_f1, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[0, 1].annotate(model.split()[0], (params[i], test_f1[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[0, 1].set_xlabel('Parameters (M)')
        axes[0, 1].set_ylabel('Test F1 Score')
        axes[0, 1].set_title('Parameters vs Performance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Training curves comparison
        for model_type, results in self.results.items():
            model_name = ABLATION_CONFIG['model_variants'][model_type]['name'].split()[0]
            axes[1, 0].plot(results['history']['val_f1'], label=model_name)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation F1')
        axes[1, 0].set_title('Training Curves Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Inference time comparison
        inference_times = [self.results[k]['inference_time'] for k in self.results.keys()]
        axes[1, 1].bar(models, inference_times, alpha=0.8, color='orange')
        axes[1, 1].set_xlabel('Model Variants')
        axes[1, 1].set_ylabel('Inference Time (ms)')
        axes[1, 1].set_title('Inference Speed Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'ablation_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function to run ablation study."""
    # Configuration
    DATA_DIR = "../CT-KIDNEY-DATASET-Normal"  # Update this path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create ablation study instance
    ablation = AblationStudy(
        data_dir=DATA_DIR,
        device=device,
        save_dir='./ablation_results'
    )
    
    # Run all experiments
    ablation.run_all_experiments()


if __name__ == "__main__":
    main()
