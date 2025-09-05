"""
Utility functions for kidney condition classification system.
Includes metrics, evaluation, threshold tuning, and explainability.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, roc_auc_score, precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

from config import CLASS_NAMES, CLASS_THRESHOLDS


def thresholded_predictions(probs: torch.Tensor) -> torch.Tensor:
    """
    Apply class-specific thresholds to model probabilities.
    
    Args:
        probs: [batch_size, num_classes] tensor with softmax probabilities.
    
    Returns:
        preds: [batch_size] tensor of predicted class indices.
    """
    batch_preds = []
    for p in probs:
        # Classes passing threshold
        passed = [
            i for i, val in enumerate(p)
            if val.item() >= CLASS_THRESHOLDS[CLASS_NAMES[i]]
        ]
        if passed:
            # pick class with highest prob among those passed
            chosen = max(passed, key=lambda i: p[i].item())
        else:
            # fallback: highest probability overall
            chosen = torch.argmax(p).item()
        batch_preds.append(chosen)
    return torch.tensor(batch_preds, dtype=torch.long, device=probs.device)


def compute_metrics(y_true, y_pred, y_probs=None, class_names=None):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (optional)
        class_names: List of class names
        
    Returns:
        metrics_dict: Dictionary containing all metrics
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics_dict = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_metrics': {}
    }
    
    # Per-class metrics
    for i, class_name in enumerate(class_names):
        metrics_dict['per_class_metrics'][class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
    
    # ROC-AUC if probabilities are provided
    if y_probs is not None:
        try:
            # Binarize labels for multiclass ROC-AUC
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
            
            # Compute ROC-AUC for each class
            roc_auc_scores = []
            for i in range(len(class_names)):
                auc_score = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
                roc_auc_scores.append(auc_score)
                metrics_dict['per_class_metrics'][class_names[i]]['roc_auc'] = auc_score
            
            metrics_dict['macro_roc_auc'] = np.mean(roc_auc_scores)
            
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")
    
    return metrics_dict


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(8, 6)):
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    return plt.gcf()


def plot_precision_recall_curves(y_true, y_probs, class_names=None, figsize=(12, 8)):
    """
    Plot precision-recall curves for each class.
    
    Args:
        y_true: Ground truth labels
        y_probs: Prediction probabilities
        class_names: List of class names
        figsize: Figure size
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    plt.figure(figsize=figsize)
    
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        ap_score = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        
        plt.plot(recall, precision, label=f'{class_name} (AP = {ap_score:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_class_distribution(class_counts, class_names=None, figsize=(10, 6)):
    """
    Plot class distribution.
    
    Args:
        class_counts: Dictionary or list of class counts
        class_names: List of class names
        figsize: Figure size
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    if isinstance(class_counts, dict):
        counts = [class_counts[name] for name in class_names]
    else:
        counts = class_counts
    
    plt.figure(figsize=figsize)
    bars = plt.bar(class_names, counts, color=['skyblue', 'lightgreen', 'lightcoral', 'orange'])
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom')
    
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()


def evaluate_model_with_thresholds(model, dataloader, device='cpu'):
    """
    Evaluate model using threshold-aware predictions.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run evaluation on
        
    Returns:
        results: Dictionary with predictions, probabilities, and targets
    """
    model.eval()
    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_confidences = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits, confidence = model(images)
            probabilities = torch.softmax(logits, dim=1)
            
            # Threshold-aware predictions
            predictions = thresholded_predictions(probabilities)
            
            # Store results
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
    
    return {
        'targets': np.array(all_targets),
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities),
        'confidences': np.array(all_confidences)
    }


class SimpleLIMEWrapper:
    """
    Simplified LIME wrapper for image explanations.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def explain_prediction(self, image, top_labels=1, num_samples=100):
        """
        Generate LIME explanation for a single image.
        
        Args:
            image: Input image tensor [1, 3, H, W]
            top_labels: Number of top predictions to explain
            num_samples: Number of samples for LIME
            
        Returns:
            explanation: Simple explanation dictionary
        """
        # This is a simplified implementation
        # In practice, you would use the lime library
        
        with torch.no_grad():
            image = image.to(self.device)
            logits, confidence = self.model(image)
            probabilities = torch.softmax(logits, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, top_labels, dim=1)
            
            explanation = {
                'top_predictions': [
                    {
                        'class': CLASS_NAMES[idx.item()],
                        'probability': prob.item(),
                        'index': idx.item()
                    }
                    for prob, idx in zip(top_probs[0], top_indices[0])
                ],
                'confidence': confidence[0].item(),
                'image_shape': image.shape
            }
            
            return explanation


def print_evaluation_report(metrics_dict):
    """
    Print formatted evaluation report.
    
    Args:
        metrics_dict: Dictionary from compute_metrics function
    """
    print("=" * 60)
    print("KIDNEY CONDITION CLASSIFICATION - EVALUATION REPORT")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics_dict['accuracy']:.4f}")
    print(f"  Macro F1: {metrics_dict['macro_f1']:.4f}")
    print(f"  Macro Precision: {metrics_dict['macro_precision']:.4f}")
    print(f"  Macro Recall: {metrics_dict['macro_recall']:.4f}")
    
    if 'macro_roc_auc' in metrics_dict:
        print(f"  Macro ROC-AUC: {metrics_dict['macro_roc_auc']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 50)
    
    for class_name, metrics in metrics_dict['per_class_metrics'].items():
        print(f"{class_name:<10} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f} "
              f"{metrics['support']:<10}")
    
    print("\nClinical Priority Assessment:")
    for class_name, metrics in metrics_dict['per_class_metrics'].items():
        priority = {
            'Tumor': 'High Recall (catch all tumors)',
            'Stone': 'Balanced Precision & Recall',
            'Cyst': 'Balanced Precision & Recall',
            'Normal': 'High Precision (avoid false alarms)'
        }.get(class_name, 'Unknown')
        
        print(f"  {class_name}: {priority}")
        print(f"    - Recall: {metrics['recall']:.4f}")
        print(f"    - Precision: {metrics['precision']:.4f}")


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 100
    n_classes = len(CLASS_NAMES)
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_probs = np.random.rand(n_samples, n_classes)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)  # Normalize
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_probs)
    print_evaluation_report(metrics)
    
    print("\nUtility functions test completed successfully!")
