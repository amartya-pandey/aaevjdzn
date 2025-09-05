"""
Evidential Deep Learning implementation for uncertainty quantification.
Based on Dirichlet distribution parameterization for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Dirichlet


def relu_evidence(y):
    """
    ReLU activation to ensure positive evidence values.
    
    Args:
        y: Raw logits from the model
        
    Returns:
        evidence: Positive evidence values (alpha - 1)
    """
    return F.relu(y)


def exp_evidence(y):
    """
    Exponential activation to ensure positive evidence values.
    More stable than ReLU for evidence computation.
    
    Args:
        y: Raw logits from the model
        
    Returns:
        evidence: Positive evidence values (alpha - 1)
    """
    return torch.exp(torch.clamp(y, -10, 10))  # Clamp to prevent overflow


def softplus_evidence(y):
    """
    Softplus activation to ensure positive evidence values.
    Smoother than ReLU, more stable than exp.
    
    Args:
        y: Raw logits from the model
        
    Returns:
        evidence: Positive evidence values (alpha - 1)
    """
    return F.softplus(y)


class EvidentialLayer(nn.Module):
    """
    Evidential layer that converts features to Dirichlet parameters.
    """
    
    def __init__(self, input_dim, num_classes, evidence_activation='exp'):
        super().__init__()
        self.num_classes = num_classes
        self.evidence_layer = nn.Linear(input_dim, num_classes)
        
        # Choose evidence activation function
        if evidence_activation == 'relu':
            self.evidence_activation = relu_evidence
        elif evidence_activation == 'exp':
            self.evidence_activation = exp_evidence
        elif evidence_activation == 'softplus':
            self.evidence_activation = softplus_evidence
        else:
            raise ValueError(f"Unknown evidence activation: {evidence_activation}")
    
    def forward(self, x):
        """
        Forward pass to compute Dirichlet parameters and uncertainty.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            dict containing:
                - logits: Raw evidence logits
                - evidence: Evidence values (alpha - 1)
                - alpha: Dirichlet concentration parameters
                - prob: Expected probabilities
                - uncertainty: Total uncertainty
                - epistemic: Epistemic uncertainty  
                - aleatoric: Aleatoric uncertainty
        """
        # Raw evidence logits
        logits = self.evidence_layer(x)
        
        # Convert to evidence (alpha - 1)
        evidence = self.evidence_activation(logits)
        
        # Dirichlet concentration parameters
        alpha = evidence + 1.0
        
        # Total evidence (precision)
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # Expected probabilities under Dirichlet
        prob = alpha / S
        
        # Uncertainty measures
        uncertainty = self.num_classes / S  # Total uncertainty
        
        # Epistemic uncertainty (vacuity)
        epistemic = uncertainty
        
        # Aleatoric uncertainty (expected entropy)
        prob_normalized = prob + 1e-10  # Avoid log(0)
        aleatoric = -torch.sum(prob_normalized * torch.log(prob_normalized), dim=1, keepdim=True)
        
        return {
            'logits': logits,
            'evidence': evidence,
            'alpha': alpha,
            'prob': prob,
            'uncertainty': uncertainty,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'precision': S
        }


class EvidentialLoss(nn.Module):
    """
    Evidential loss function for Dirichlet-based uncertainty quantification.
    Combines likelihood loss with KL divergence regularization.
    """
    
    def __init__(self, num_classes, annealing_step=10, lambda_reg=1e-2):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.lambda_reg = lambda_reg
        
    def kl_divergence(self, alpha, target):
        """
        Compute KL divergence between Dirichlet distributions.
        KL[Dir(alpha) || Dir(1)] where Dir(1) is uniform prior.
        
        Args:
            alpha: Dirichlet concentration parameters [batch_size, num_classes]
            target: One-hot encoded targets [batch_size, num_classes]
            
        Returns:
            kl_div: KL divergence values [batch_size]
        """
        # Prior: uniform Dirichlet Dir(1, 1, ..., 1)
        beta = torch.ones_like(alpha)
        
        # KL divergence computation
        S_alpha = torch.sum(alpha, dim=1)
        S_beta = torch.sum(beta, dim=1)
        
        lnB_alpha = torch.sum(torch.lgamma(alpha), dim=1) - torch.lgamma(S_alpha)
        lnB_beta = torch.sum(torch.lgamma(beta), dim=1) - torch.lgamma(S_beta)
        
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        
        kl = lnB_alpha - lnB_beta + torch.sum((alpha - beta) * (dg1 - dg0.unsqueeze(1)), dim=1)
        
        return kl
    
    def loglikelihood_loss(self, alpha, target):
        """
        Compute negative log-likelihood under Dirichlet distribution.
        
        Args:
            alpha: Dirichlet concentration parameters [batch_size, num_classes]
            target: One-hot encoded targets [batch_size, num_classes]
            
        Returns:
            likelihood_loss: Negative log-likelihood [batch_size]
        """
        S = torch.sum(alpha, dim=1)
        loglikelihood = torch.sum(target * (torch.digamma(S.unsqueeze(1)) - torch.digamma(alpha)), dim=1)
        return loglikelihood
    
    def forward(self, evidential_output, targets, epoch=None):
        """
        Compute evidential loss.
        
        Args:
            evidential_output: Output from EvidentialLayer
            targets: Ground truth labels [batch_size]
            epoch: Current training epoch for annealing
            
        Returns:
            total_loss: Combined evidential loss
            loss_dict: Dictionary with loss components
        """
        if 'alpha' not in evidential_output:
            raise ValueError("evidential_output must contain 'alpha' key for EvidentialLoss computation")
        
        alpha = evidential_output['alpha']
        batch_size = alpha.size(0)
        
        # Convert targets to one-hot
        target_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Likelihood loss (maximize evidence for correct class)
        likelihood_loss = self.loglikelihood_loss(alpha, target_one_hot)
        likelihood_loss = torch.mean(likelihood_loss)
        
        # KL divergence regularization (penalize overconfidence)
        kl_div = self.kl_divergence(alpha, target_one_hot)
        
        # Annealing factor for KL regularization
        if epoch is not None:
            annealing_coef = min(1.0, epoch / self.annealing_step)
        else:
            annealing_coef = 1.0
        
        kl_loss = annealing_coef * torch.mean(kl_div)
        
        # Total evidential loss
        total_loss = likelihood_loss + self.lambda_reg * kl_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'likelihood_loss': likelihood_loss.item(),
            'kl_loss': kl_loss.item(),
            'annealing_coef': annealing_coef
        }
        
        return total_loss, loss_dict


def compute_evidential_metrics(evidential_output, targets):
    """
    Compute additional metrics for evidential predictions.
    
    Args:
        evidential_output: Output from EvidentialLayer or dict with at least 'prob' and 'uncertainty'
        targets: Ground truth labels
        
    Returns:
        metrics_dict: Dictionary with evidential metrics
    """
    # Safely extract required keys
    if 'prob' not in evidential_output:
        raise ValueError("evidential_output must contain 'prob' key")
    if 'uncertainty' not in evidential_output:
        raise ValueError("evidential_output must contain 'uncertainty' key")
    
    prob = evidential_output['prob']
    uncertainty = evidential_output['uncertainty']
    
    # Handle optional keys for backward compatibility
    epistemic = evidential_output.get('epistemic', uncertainty)  # Use total uncertainty if epistemic not available
    aleatoric = evidential_output.get('aleatoric', torch.zeros_like(uncertainty))  # Default to zero if not available
    alpha = evidential_output.get('alpha', None)
    
    # Predicted classes
    predicted = torch.argmax(prob, dim=1)
    
    # Accuracy
    accuracy = (predicted == targets).float().mean().item()
    
    # Average uncertainties
    avg_uncertainty = uncertainty.mean().item()
    avg_epistemic = epistemic.mean().item()
    avg_aleatoric = aleatoric.mean().item()
    
    # Confidence (max probability)
    max_prob = torch.max(prob, dim=1)[0]
    avg_confidence = max_prob.mean().item()
    
    # Precision (total evidence) - only if alpha is available
    if alpha is not None:
        precision = torch.sum(alpha, dim=1)
        avg_precision = precision.mean().item()
    else:
        avg_precision = 0.0  # Default value when alpha not available
    
    # Uncertainty for correct vs incorrect predictions
    correct_mask = (predicted == targets)
    if correct_mask.any():
        uncertainty_correct = uncertainty[correct_mask].mean().item()
    else:
        uncertainty_correct = 0.0
        
    if (~correct_mask).any():
        uncertainty_incorrect = uncertainty[~correct_mask].mean().item()
    else:
        uncertainty_incorrect = 0.0
    
    metrics_dict = {
        'accuracy': accuracy,
        'avg_uncertainty': avg_uncertainty,
        'avg_epistemic': avg_epistemic,
        'avg_aleatoric': avg_aleatoric,
        'avg_confidence': avg_confidence,
        'avg_precision': avg_precision,
        'uncertainty_correct': uncertainty_correct,
        'uncertainty_incorrect': uncertainty_incorrect,
    }
    
    return metrics_dict


if __name__ == "__main__":
    # Test evidential components
    batch_size, input_dim, num_classes = 8, 512, 4
    
    # Create test data
    features = torch.randn(batch_size, input_dim)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test evidential layer
    evidential_layer = EvidentialLayer(input_dim, num_classes)
    output = evidential_layer(features)
    
    print("Evidential Layer Output:")
    for key, value in output.items():
        print(f"{key}: {value.shape}")
    
    # Test evidential loss
    loss_fn = EvidentialLoss(num_classes)
    total_loss, loss_dict = loss_fn(output, targets, epoch=5)
    
    print(f"\nEvidential Loss: {total_loss:.4f}")
    print("Loss components:", loss_dict)
    
    # Test metrics
    metrics = compute_evidential_metrics(output, targets)
    print("\nEvidential Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
