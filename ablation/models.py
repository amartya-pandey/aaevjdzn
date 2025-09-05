"""
Dual-path kidney condition classification model with cross-attention fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import MODEL_CONFIG, NUM_CLASSES


class CrossAttentionFusion(nn.Module):
    """Cross-attention block for fusing features from two pathways."""
    
    def __init__(self, feature_dim, attention_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.query_proj = nn.Linear(feature_dim, attention_dim)
        self.key_proj = nn.Linear(feature_dim, attention_dim)
        self.value_proj = nn.Linear(feature_dim, attention_dim)
        
        # Output projection
        self.output_proj = nn.Linear(attention_dim, feature_dim)
        self.dropout = nn.Dropout(MODEL_CONFIG['dropout_rate'])
        
    def forward(self, path_a_features, path_b_features):
        """
        Args:
            path_a_features: [batch_size, feature_dim] - EfficientNet features
            path_b_features: [batch_size, feature_dim] - Custom CNN features
        """
        batch_size = path_a_features.size(0)
        
        # Prepare queries, keys, values
        q_a = self.query_proj(path_a_features)  # [batch_size, attention_dim]
        k_b = self.key_proj(path_b_features)    # [batch_size, attention_dim]
        v_b = self.value_proj(path_b_features)  # [batch_size, attention_dim]
        
        q_b = self.query_proj(path_b_features)  # [batch_size, attention_dim]
        k_a = self.key_proj(path_a_features)    # [batch_size, attention_dim]
        v_a = self.value_proj(path_a_features)  # [batch_size, attention_dim]
        
        # Cross-attention: A attends to B, B attends to A
        attention_a_to_b = F.softmax(torch.sum(q_a * k_b, dim=1, keepdim=True), dim=1)
        attention_b_to_a = F.softmax(torch.sum(q_b * k_a, dim=1, keepdim=True), dim=1)
        
        # Apply attention
        attended_a = attention_a_to_b * v_b
        attended_b = attention_b_to_a * v_a
        
        # Combine attended features
        fused_features = self.output_proj(attended_a + attended_b)
        fused_features = self.dropout(fused_features)
        
        return fused_features


class LightweightCNN(nn.Module):
    """Lightweight CNN for local feature extraction."""
    
    def __init__(self, input_channels=3, feature_dim=512):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))



        )
        
        self.fc = nn.Linear(256, feature_dim)
        self.dropout = nn.Dropout(MODEL_CONFIG['dropout_rate'])
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = self.dropout(x)
        return x
    
    
class DualPathKidneyClassifier(nn.Module):
    """
    Dual-path model for kidney condition classification.
    Combines EfficientNet backbone with lightweight CNN using cross-attention.
    """
    
    def __init__(self):
        super().__init__()
        
        # Path A: EfficientNet backbone (global features)
        self.efficientnet = timm.create_model(
            MODEL_CONFIG['efficientnet_model'], 
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Get EfficientNet feature dimension
        efficientnet_features = self.efficientnet.num_features
        
        # Project EfficientNet features to common dimension
        self.efficientnet_proj = nn.Linear(efficientnet_features, MODEL_CONFIG['feature_dim'])
        
        # Path B: Lightweight CNN (local features)
        self.lightweight_cnn = LightweightCNN(
            input_channels=3, 
            feature_dim=MODEL_CONFIG['feature_dim']
        )
        
        # Cross-attention fusion
        self.cross_attention = CrossAttentionFusion(
            feature_dim=MODEL_CONFIG['feature_dim'],
            attention_dim=MODEL_CONFIG['attention_dim']
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(MODEL_CONFIG['feature_dim'], MODEL_CONFIG['feature_dim'] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(MODEL_CONFIG['feature_dim'] // 2, NUM_CLASSES)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(MODEL_CONFIG['feature_dim'], MODEL_CONFIG['feature_dim'] // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(MODEL_CONFIG['feature_dim'] // 4, 1),
            nn.Sigmoid()  # Output confidence between 0 and 1
        )
        
    def forward(self, x):
        """
        Forward pass through dual-path architecture.
        
        Args:
            x: Input tensor [batch_size, 3, height, width]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
            confidence: Confidence scores [batch_size, 1]
        """
        # Path A: EfficientNet features
        efficientnet_features = self.efficientnet(x)
        efficientnet_features = self.efficientnet_proj(efficientnet_features)
        
        # Path B: Lightweight CNN features
        cnn_features = self.lightweight_cnn(x)
        
        # Cross-attention fusion
        fused_features = self.cross_attention(efficientnet_features, cnn_features)
        
        # Classification and confidence prediction
        logits = self.classifier(fused_features)
        confidence = self.confidence_head(fused_features)
        
        return logits, confidence


    def freeze_for_dual_path(self, freeze_stages=[0, 1, 2, 3]):
        """
        Freeze EfficientNet layers optimally for dual-path architecture.
        Since CNN handles local features, freeze more EfficientNet early layers.
        
        Args:
            freeze_stages: List of stage indices to freeze (default: [0,1,2,3])
        """
        frozen_params = 0
        trainable_params = 0
        
        # Freeze stem (basic convolutions)
        if hasattr(self.efficientnet, 'conv_stem'):
            for param in self.efficientnet.conv_stem.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        if hasattr(self.efficientnet, 'bn1'):
            for param in self.efficientnet.bn1.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
        
        # Freeze specified stages
        if hasattr(self.efficientnet, 'blocks'):
            for stage_idx, block in enumerate(self.efficientnet.blocks):
                if stage_idx in freeze_stages:
                    for param in block.parameters():
                        param.requires_grad = False
                        frozen_params += param.numel()
                    print(f"❄️ Frozen EfficientNet stage {stage_idx}")
                else:
                    for param in block.parameters():
                        param.requires_grad = True
                        trainable_params += param.numel()
                    print(f"🔥 Trainable EfficientNet stage {stage_idx}")
        
        # Always keep head trainable
        if hasattr(self.efficientnet, 'conv_head'):
            for param in self.efficientnet.conv_head.parameters():
                param.requires_grad = True
                trainable_params += param.numel()
        
        if hasattr(self.efficientnet, 'classifier') and self.efficientnet.classifier is not None:
            for param in self.efficientnet.classifier.parameters():
                param.requires_grad = True
                trainable_params += param.numel()
        
        print(f"\n📊 EfficientNet Freezing Summary:")
        print(f"   Frozen: {frozen_params:,} parameters")
        print(f"   Trainable: {trainable_params:,} parameters")
        print(f"   Frozen ratio: {frozen_params/(frozen_params+trainable_params)*100:.1f}%")
        
        return frozen_params, trainable_params
    
    def unfreeze_gradually(self, epoch, total_epochs, unfreeze_schedule=0.6):
        """
        Gradually unfreeze EfficientNet layers during training.
        
        Args:
            epoch: Current epoch
            total_epochs: Total training epochs
            unfreeze_schedule: Fraction of training when to start unfreezing (default: 0.6)
        """
        if epoch > total_epochs * unfreeze_schedule:
            # Progressive unfreezing: unfreeze one more stage every few epochs
            stages_to_unfreeze = min(3, (epoch - int(total_epochs * unfreeze_schedule)) // 3)
            
            if hasattr(self.efficientnet, 'blocks'):
                for stage_idx in range(stages_to_unfreeze):
                    if stage_idx < len(self.efficientnet.blocks):
                        for param in self.efficientnet.blocks[stage_idx].parameters():
                            param.requires_grad = True
                        print(f"🔓 Unfroze EfficientNet stage {stage_idx} at epoch {epoch}")
    
    def get_parameter_groups(self, base_lr=1e-4):
        """
        Get parameter groups with different learning rates for different components.
        
        Args:
            base_lr: Base learning rate
            
        Returns:
            List of parameter groups for optimizer
        """
        # EfficientNet parameters (frozen and unfrozen)
        efficientnet_frozen = []
        efficientnet_trainable = []
        
        for name, param in self.efficientnet.named_parameters():
            if param.requires_grad:
                efficientnet_trainable.append(param)
            else:
                efficientnet_frozen.append(param)
        
        # CNN parameters (always trainable)
        cnn_params = list(self.lightweight_cnn.parameters())
        
        # Other components (projection, attention, heads)
        other_params = (
            list(self.efficientnet_proj.parameters()) +
            list(self.cross_attention.parameters()) +
            list(self.classifier.parameters()) +
            list(self.confidence_head.parameters())
        )
        
        param_groups = [
            {
                'params': efficientnet_trainable,
                'lr': base_lr * 0.1,  # Lower LR for pretrained layers
                'name': 'efficientnet_trainable'
            },
            {
                'params': cnn_params,
                'lr': base_lr,  # Normal LR for CNN
                'name': 'cnn'
            },
            {
                'params': other_params,
                'lr': base_lr,  # Normal LR for new components
                'name': 'other'
            }
        ]
        
        # Filter out empty parameter groups
        param_groups = [group for group in param_groups if len(group['params']) > 0]
        
        print(f"Parameter groups created:")
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            print(f"  {group['name']}: {num_params:,} parameters, LR: {group['lr']}")
        
        return param_groups
    
    def print_freezing_summary(self):
        """Print summary of frozen vs trainable parameters."""
        frozen_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
        
        total_params = frozen_params + trainable_params
        
        print(f"\n🧊 Model Freezing Summary:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        
        # Component breakdown
        print(f"\n📋 Component Breakdown:")
        
        # EfficientNet
        eff_frozen = sum(p.numel() for name, p in self.efficientnet.named_parameters() if not p.requires_grad)
        eff_trainable = sum(p.numel() for name, p in self.efficientnet.named_parameters() if p.requires_grad)
        print(f"   EfficientNet: {eff_frozen:,} frozen, {eff_trainable:,} trainable")
        
        # CNN
        cnn_params = sum(p.numel() for p in self.lightweight_cnn.parameters())
        print(f"   Custom CNN: {cnn_params:,} trainable")
        
        # Other components
        other_params = (
            sum(p.numel() for p in self.efficientnet_proj.parameters()) +
            sum(p.numel() for p in self.cross_attention.parameters()) +
            sum(p.numel() for p in self.classifier.parameters()) +
            sum(p.numel() for p in self.confidence_head.parameters())
        )
        print(f"   Other components: {other_params:,} trainable")
        
        return {
            'total': total_params,
            'frozen': frozen_params,
            'trainable': trainable_params,
            'efficientnet_frozen': eff_frozen,
            'efficientnet_trainable': eff_trainable,
            'cnn': cnn_params,
            'other': other_params
        }
    

def create_model(model_type='full'):
    """
    Factory function to create different model variants for ablation studies.
    
    Args:
        model_type (str): Type of model to create
            - 'full': Full dual-path model with cross-attention (default)
            - 'efficientnet_only': EfficientNet-B4 only (Path A)
            - 'cnn_only': Lightweight CNN only (Path B)
            - 'simple_fusion': Dual-path with simple concatenation fusion
    
    Returns:
        Model instance
    """
    if model_type == 'full':
        return DualPathKidneyClassifier()
    elif model_type == 'efficientnet_only':
        return EfficientNetOnly()
    elif model_type == 'cnn_only':
        return CNNOnly()
    elif model_type == 'simple_fusion':
        return DualPathSimpleFusion()
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                        f"Choose from: 'full', 'efficientnet_only', 'cnn_only', 'simple_fusion'")

# Alias for backward compatibility
DualPathKidneyModel = DualPathKidneyClassifier


if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_model()
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        logits, confidence = model(dummy_input)
        print(f"Logits shape: {logits.shape}")
        print(f"Confidence shape: {confidence.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# Ablation Study Models

class EfficientNetOnly(nn.Module):
    """
    Ablation Study 1: EfficientNet-B4 only (Path A only)
    Tests the performance of global features without local CNN features.
    """
    
    def __init__(self):
        super().__init__()
        
        # EfficientNet backbone only
        self.efficientnet = timm.create_model(
            MODEL_CONFIG['efficientnet_model'], 
            pretrained=True,
            num_classes=0
        )
        
        # Get EfficientNet feature dimension
        efficientnet_features = self.efficientnet.num_features
        
        # Direct classification head
        self.classifier = nn.Sequential(
            nn.Linear(efficientnet_features, MODEL_CONFIG['feature_dim']),
            nn.ReLU(inplace=True),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(MODEL_CONFIG['feature_dim'], MODEL_CONFIG['feature_dim'] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(MODEL_CONFIG['feature_dim'] // 2, NUM_CLASSES)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(efficientnet_features, MODEL_CONFIG['feature_dim'] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(MODEL_CONFIG['feature_dim'] // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Only EfficientNet path
        features = self.efficientnet(x)
        logits = self.classifier(features)
        confidence = self.confidence_head(features)
        return logits, confidence


class CNNOnly(nn.Module):
    """
    Ablation Study 2: Lightweight CNN only (Path B only)
    Tests the performance of local features without EfficientNet.
    """
    
    def __init__(self):
        super().__init__()
        
        # Only lightweight CNN
        self.lightweight_cnn = LightweightCNN(
            input_channels=3, 
            feature_dim=MODEL_CONFIG['feature_dim']
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(MODEL_CONFIG['feature_dim'], MODEL_CONFIG['feature_dim'] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(MODEL_CONFIG['feature_dim'] // 2, NUM_CLASSES)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(MODEL_CONFIG['feature_dim'], MODEL_CONFIG['feature_dim'] // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(MODEL_CONFIG['feature_dim'] // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Only CNN path
        features = self.lightweight_cnn(x)
        logits = self.classifier(features)
        confidence = self.confidence_head(features)
        return logits, confidence


class DualPathSimpleFusion(nn.Module):
    """
    Ablation Study 3: Dual-path with simple concatenation fusion
    Tests dual-path architecture with basic fusion instead of cross-attention.
    """
    
    def __init__(self):
        super().__init__()
        
        # Path A: EfficientNet backbone
        self.efficientnet = timm.create_model(
            MODEL_CONFIG['efficientnet_model'], 
            pretrained=True,
            num_classes=0
        )
        
        # Get EfficientNet feature dimension
        efficientnet_features = self.efficientnet.num_features
        
        # Project EfficientNet features to common dimension
        self.efficientnet_proj = nn.Linear(efficientnet_features, MODEL_CONFIG['feature_dim'])
        
        # Path B: Lightweight CNN
        self.lightweight_cnn = LightweightCNN(
            input_channels=3, 
            feature_dim=MODEL_CONFIG['feature_dim']
        )
        
        # Simple fusion layer (concatenation + projection)
        self.fusion_layer = nn.Sequential(
            nn.Linear(MODEL_CONFIG['feature_dim'] * 2, MODEL_CONFIG['feature_dim']),
            nn.ReLU(inplace=True),
            nn.Dropout(MODEL_CONFIG['dropout_rate'])
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(MODEL_CONFIG['feature_dim'], MODEL_CONFIG['feature_dim'] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(MODEL_CONFIG['feature_dim'] // 2, NUM_CLASSES)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(MODEL_CONFIG['feature_dim'], MODEL_CONFIG['feature_dim'] // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(MODEL_CONFIG['feature_dim'] // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Path A: EfficientNet features
        efficientnet_features = self.efficientnet(x)
        efficientnet_features = self.efficientnet_proj(efficientnet_features)
        
        # Path B: CNN features
        cnn_features = self.lightweight_cnn(x)
        
        # Simple fusion: concatenation
        fused_features = torch.cat([efficientnet_features, cnn_features], dim=1)
        fused_features = self.fusion_layer(fused_features)
        
        # Classification and confidence prediction
        logits = self.classifier(fused_features)
        confidence = self.confidence_head(fused_features)
        
        return logits, confidence
