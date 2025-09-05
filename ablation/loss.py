"""
Combined loss functions for the dual-path kidney classification model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CLASS_WEIGHTS, CLASS_NAMES, MODEL_CONFIG


class CombinedLoss(nn.Module):
    """
    Combined loss function with weighted cross-entropy and confidence calibration.
    """
    
    def __init__(self, confidence_weight=None):
        super().__init__()
        
        # Set up class weights for cross-entropy loss
        weights = torch.tensor([CLASS_WEIGHTS[name] for name in CLASS_NAMES], dtype=torch.float32)
        self.register_buffer('class_weights', weights)
        
        # Confidence loss weight
        self.confidence_weight = confidence_weight or MODEL_CONFIG['confidence_loss_weight']
        
        # Loss functions - don't pass weights in constructor, will pass in forward
        self.mse_loss = nn.MSELoss()
        
    def forward(self, logits, confidence, targets):
        """
        Compute combined loss.
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            confidence: Confidence scores [batch_size, 1]
            targets: Ground truth labels [batch_size]
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Classification loss (weighted cross-entropy) - use F.cross_entropy with weights
        classification_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        
        # Confidence calibration loss
        # Target confidence should be high when prediction is correct
        predicted_classes = torch.argmax(logits, dim=1)
        correct_predictions = (predicted_classes == targets).float().unsqueeze(1)
        confidence_loss = self.mse_loss(confidence, correct_predictions)
        
        # Combined loss
        total_loss = classification_loss + self.confidence_weight * confidence_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'classification_loss': classification_loss.item(),
            'confidence_loss': confidence_loss.item()
        }
        
        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Alternative to weighted cross-entropy.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        
        if alpha is None:
            # Use class weights as alpha
            alpha = torch.tensor([CLASS_WEIGHTS[name] for name in CLASS_NAMES], dtype=torch.float32)
        
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            loss: Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedFocalLoss(nn.Module):
    """
    Combined loss with Focal Loss and confidence calibration.
    """
    
    def __init__(self, alpha=None, gamma=2.0, confidence_weight=None):
        super().__init__()
        
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.confidence_weight = confidence_weight or MODEL_CONFIG['confidence_loss_weight']
        self.mse_loss = nn.MSELoss()
        
    def forward(self, logits, confidence, targets):
        """
        Compute combined focal loss.
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            confidence: Confidence scores [batch_size, 1]
            targets: Ground truth labels [batch_size]
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Focal loss for classification
        classification_loss = self.focal_loss(logits, targets)
        
        # Confidence calibration loss
        predicted_classes = torch.argmax(logits, dim=1)
        correct_predictions = (predicted_classes == targets).float().unsqueeze(1)
        confidence_loss = self.mse_loss(confidence, correct_predictions)
        
        # Combined loss
        total_loss = classification_loss + self.confidence_weight * confidence_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'classification_loss': classification_loss.item(),
            'confidence_loss': confidence_loss.item()
        }
        
        return total_loss, loss_dict


def create_loss_function(loss_type='combined'):
    """
    Factory function to create loss function.
    
    Args:
        loss_type: Type of loss ('combined', 'focal')
        
    Returns:
        loss_fn: Loss function instance
    """
    if loss_type == 'combined':
        return CombinedLoss()
    elif loss_type == 'focal':
        return CombinedFocalLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    batch_size = 8
    num_classes = 4
    
    # Dummy data
    logits = torch.randn(batch_size, num_classes)
    confidence = torch.rand(batch_size, 1)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test combined loss
    loss_fn = create_loss_function('combined')
    total_loss, loss_dict = loss_fn(logits, confidence, targets)
    
    print(f"Combined Loss: {total_loss:.4f}")
    print(f"Loss components: {loss_dict}")
    
    # Test focal loss
    focal_loss_fn = create_loss_function('focal')
    total_loss_focal, loss_dict_focal = focal_loss_fn(logits, confidence, targets)
    
    print(f"Focal Loss: {total_loss_focal:.4f}")
    print(f"Loss components: {loss_dict_focal}")
