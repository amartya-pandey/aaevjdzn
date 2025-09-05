"""
Configuration file for kidney condition classification system.
Contains dataset stats, class weights, thresholds, and hyperparameters.
"""

# Class definitions
CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']
NUM_CLASSES = len(CLASS_NAMES)

# Dataset statistics
DATASET_STATS = {
    'total_samples': 12446,
    'class_distribution': {
        'Cyst': 3709,
        'Normal': 5077,
        'Stone': 1377,
        'Tumor': 2283
    }
}

# Class weights (based on dataset distribution)
CLASS_WEIGHTS = {
    'Cyst': 0.84,
    'Normal': 0.61,
    'Stone': 2.26,
    'Tumor': 1.36
}

# Class-specific thresholds for clinical priorities
CLASS_THRESHOLDS = {
    'Cyst': 0.50,    # Balanced precision & recall
    'Normal': 0.60,  # Prioritize precision (avoid false alarms)
    'Stone': 0.50,   # Balance precision & recall
    'Tumor': 0.35    # Maximize recall (catch all cases)
}

# Model hyperparameters
MODEL_CONFIG = {
    # 'efficientnet_model': 'efficientnet_b0',  # for b0
    # 'image_size': 224,    # for b0
    'efficientnet_model': 'efficientnet_b4',  # for b4
    'input_size': 380,  # for b4
    'feature_dim': 512,
    'attention_dim': 256,
    'dropout_rate': 0.3,
    'confidence_loss_weight': 0.1,
    'scheduler_step_size': 10,
    'scheduler_gamma': 0.5
}

# Training hyperparameters
# TRAINING_CONFIG = { # for b0
#     'batch_size': 32,
#     'learning_rate': 1e-4,
#     'num_epochs': 50,
#     'patience': 10,
#     'weight_decay': 1e-5,
#     'scheduler_step_size': 10,
#     'scheduler_gamma': 0.5
# }

TRAINING_CONFIG = {     # for b4
    'batch_size': 16,  # will reduce it for B4's memory requirements
    'learning_rate': 8e-5,  # Slightly reduced for larger model stability
    'num_epochs': 50,
    'patience': 12,  # Increased patience for larger model
    'min_delta': 0.001,
    'warmup_epochs': 5,
    'weight_decay': 2e-4,  # Increased regularization
    'gradient_clip_norm': 1.0
}
# Data preprocessing
DATA_CONFIG = {
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'augmentation': {
        'rotation_degrees': 15,
        'brightness': 0.1,
        'contrast': 0.1,
        'saturation': 0.1,
        'hue': 0.05
    }
}

# Transfer learning and freezing configuration
FREEZING_CONFIG = {
    'enabled': True,                    # Enable freezing for dual-path training
    'freeze_stages': [0, 1, 2, 3],     # EfficientNet stages to freeze (0-3 recommended)
    'gradual_unfreeze': True,           # Enable gradual unfreezing during training
    'unfreeze_start_ratio': 0.6,       # Start unfreezing at 60% of training
    'differential_lr': True,            # Use different learning rates for components
    'efficientnet_lr_multiplier': 0.1,  # LR multiplier for unfrozen EfficientNet layers
}

# Ablation study configuration
ABLATION_CONFIG = {
    'model_variants': {
        'full': {
            'name': 'Full Dual-Path with Cross-Attention',
            'description': 'Complete model with EfficientNet + CNN + Cross-attention fusion',
            'model_type': 'full'
        },
        'efficientnet_only': {
            'name': 'EfficientNet-B4 Only (Path A)',
            'description': 'Only EfficientNet backbone for global feature extraction',
            'model_type': 'efficientnet_only'
        },
        'cnn_only': {
            'name': 'Lightweight CNN Only (Path B)',
            'description': 'Only custom CNN for local feature extraction',
            'model_type': 'cnn_only'
        },
        'simple_fusion': {
            'name': 'Dual-Path with Simple Fusion',
            'description': 'Both paths with concatenation fusion instead of cross-attention',
            'model_type': 'simple_fusion'
        }
    },
    'comparison_metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'auc_roc',
        'parameters_count',
        'inference_time'
    ]
}

# Clinical priorities mapping
CLINICAL_PRIORITIES = {
    'Tumor': 'maximize_recall',      # Catch all tumors
    'Stone': 'balance_precision_recall',  # Balanced approach
    'Cyst': 'balance_precision_recall',   # Balanced approach
    'Normal': 'prioritize_precision'      # Avoid false alarms
}
