"""
Data loading and preprocessing for kidney condition classification.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import DATA_CONFIG, MODEL_CONFIG, CLASS_NAMES


class KidneyDataset(Dataset):
    """Dataset class for kidney ultrasound images."""
    
    def __init__(self, image_paths, labels, transform=None, class_to_idx=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (strings or integers)
            transform: Optional transform to be applied on images
            class_to_idx: Dictionary mapping class names to indices
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Create class to index mapping if not provided
        if class_to_idx is None:
            self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        else:
            self.class_to_idx = class_to_idx
        
        # Convert string labels to indices if necessary
        if len(labels) > 0 and isinstance(labels[0], str):
            self.labels = [self.class_to_idx[label] for label in labels]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Get distribution of classes in the dataset."""
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = {}
        for idx, count in zip(unique, counts):
            class_name = CLASS_NAMES[idx]
            distribution[class_name] = count
        return distribution


def get_transforms(split='train'):
    """
    Get image transforms for different splits.
    
    Args:
        split: 'train', 'val', or 'test'
        
    Returns:
        transform: Composed transforms
    """
    image_size = MODEL_CONFIG['input_size']
    
    if split == 'train':
        # Training transforms with augmentation
        transform = transforms.Compose([
            # transforms.Resize((image_size + 32, image_size + 32)), # for b0
            transforms.Resize((image_size + 36, image_size + 36)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=DATA_CONFIG['augmentation']['rotation_degrees']),
            transforms.ColorJitter(
                brightness=DATA_CONFIG['augmentation']['brightness'],
                contrast=DATA_CONFIG['augmentation']['contrast'],
                saturation=DATA_CONFIG['augmentation']['saturation'],
                hue=DATA_CONFIG['augmentation']['hue']
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def load_kidney_data(data_dir):
    """
    Load kidney ultrasound data from directory structure.
    
    Expected structure:
    data_dir/
    ├── CT-KIDNEY-DATASET-Normal/
    │   ├── CT-KIDNEY-DATASET-Normal/
    │   │   ├── Cyst/           (images)
    │   │   ├── Normal/         (images)
    │   │   ├── Stone/          (images)
    │   │   └── Tumor/          (images)
    │   └── kidneyData.csv
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        image_paths: List of image paths
        labels: List of corresponding labels
    """
    data_path = Path(data_dir)
    
    # Look for the nested CT-KIDNEY-DATASET-Normal directory structure
    dataset_dir = data_path
    
    if not dataset_dir.exists():
        # Try alternative structure (direct CT-KIDNEY-DATASET-Normal)
        dataset_dir = data_path / "CT-KIDNEY-DATASET-Normal"
        if not dataset_dir.exists():
            # Try the data_dir itself if it contains class folders
            dataset_dir = data_path
    
    image_paths = []
    labels = []
    found_classes = []
    
    # Check which class directories exist
    for class_name in CLASS_NAMES:
        class_dir = dataset_dir / class_name
        if class_dir.exists() and class_dir.is_dir():
            found_classes.append(class_name)
    
    if not found_classes:
        raise FileNotFoundError(
            f"Could not find any class directories in {dataset_dir}. "
            f"Expected directories: {CLASS_NAMES}"
        )
    
    print(f"Found class directories: {found_classes}")
    
    # Define supported image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG']
    
    # Load images from each class directory
    for class_name in found_classes:
        class_dir = dataset_dir / class_name
        
        # Get all image files in the class directory
        class_images = []
        
        for ext in image_extensions:
            class_images.extend(list(class_dir.glob(ext)))
        
        # Add images to the lists
        for image_file in class_images:
            image_paths.append(str(image_file))
            labels.append(class_name)
        
        print(f"  {class_name}: {len(class_images)} images")
    
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {dataset_dir}. "
            f"Supported formats: {image_extensions}"
        )
    
    print(f"\nTotal loaded: {len(image_paths)} images from {len(set(labels))} classes")
    
    return image_paths, labels


def create_data_splits(image_paths, labels, test_size=None, val_size=None, random_state=42):
    """
    Create train/validation/test splits.
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        test_size: Test split ratio
        val_size: Validation split ratio
        random_state: Random seed
        
    Returns:
        splits: Dictionary with train/val/test splits
    """
    if test_size is None:
        test_size = DATA_CONFIG['test_split']
    if val_size is None:
        val_size = DATA_CONFIG['val_split']
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_size, 
        stratify=labels, random_state=random_state
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for smaller remaining set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted,
        stratify=y_temp, random_state=random_state
    )
    
    splits = {
        'train': {'paths': X_train, 'labels': y_train},
        'val': {'paths': X_val, 'labels': y_val},
        'test': {'paths': X_test, 'labels': y_test}
    }
    
    # Print split statistics
    print(f"\nData splits:")
    for split_name, split_data in splits.items():
        print(f"  {split_name}: {len(split_data['paths'])} samples")
        unique, counts = np.unique(split_data['labels'], return_counts=True)
        for label, count in zip(unique, counts):
            print(f"    {label}: {count}")
    
    return splits


def create_datasets(data_dir):
    """
    Create train, validation, and test datasets.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        datasets: Dictionary with train/val/test datasets
    """
    # Load data
    image_paths, labels = load_kidney_data(data_dir)
    
    # Create splits
    splits = create_data_splits(image_paths, labels)
    
    # Create datasets with appropriate transforms
    datasets = {}
    for split_name, split_data in splits.items():
        transform = get_transforms(split_name if split_name == 'train' else 'val')
        datasets[split_name] = KidneyDataset(
            image_paths=split_data['paths'],
            labels=split_data['labels'],
            transform=transform
        )
    
    return datasets


def create_dataloaders(datasets, batch_size=None):
    """
    Create dataloaders for train, validation, and test datasets.
    
    Args:
        datasets: Dictionary with datasets
        batch_size: Batch size for dataloaders
        
    Returns:
        dataloaders: Dictionary with dataloaders
    """
    if batch_size is None:
        batch_size = DATA_CONFIG.get('batch_size', 32)
    
    dataloaders = {}
    
    for split_name, dataset in datasets.items():
        shuffle = (split_name == 'train')
        
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split_name == 'train')  # Drop last batch for training only
        )
    
    return dataloaders


def analyze_dataset(data_dir):
    """
    Analyze the dataset and print statistics.
    
    Args:
        data_dir: Path to data directory
    """
    print("Analyzing kidney ultrasound dataset...")
    
    try:
        image_paths, labels = load_kidney_data(data_dir)
        
        print(f"\nDataset Statistics:")
        print(f"  Total images: {len(image_paths)}")
        print(f"  Number of classes: {len(set(labels))}")
        
        # Class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\nClass Distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = 100 * count / len(labels)
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Check for class imbalance
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2:
            print("  Warning: Significant class imbalance detected!")
            print("  Consider using weighted sampling or class weights.")
        
        # Check if all expected classes are present
        missing_classes = set(CLASS_NAMES) - set(unique_labels)
        if missing_classes:
            print(f"\nMissing classes: {missing_classes}")
        else:
            print(f"\nAll expected classes found: {CLASS_NAMES}")
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        print(f"Please ensure the dataset follows this structure:")
        print(f"  {data_dir}/")
        print(f"  ├── CT-KIDNEY-DATASET-Normal/")
        print(f"  │   ├── CT-KIDNEY-DATASET-Normal/")
        print(f"  │   │   ├── Cyst/           (images)")
        print(f"  │   │   ├── Normal/         (images)")
        print(f"  │   │   ├── Stone/          (images)")
        print(f"  │   │   └── Tumor/          (images)")
        print(f"  │   └── kidneyData.csv")
        print(f"  OR simply place class folders directly in:")
        print(f"  {data_dir}/Cyst/, {data_dir}/Normal/, etc.")


if __name__ == "__main__":
    # Test data loading functions
    print("Testing data loading module...")
    
    # Example usage (would require actual data directory):
    # analyze_dataset("/path/to/CT-KIDNEY-DATASET-Normal")
    
    # Test transforms
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    print(f"Train transform: {train_transform}")
    print(f"Val transform: {val_transform}")
    
    print("Data loading module loaded successfully!")
    print(f"Supported classes: {CLASS_NAMES}")
    print(f"Data configuration: {DATA_CONFIG}")
    print(f"\nExpected directory structure:")
    print(f"  data_dir/")
    print(f"  ├── CT-KIDNEY-DATASET-Normal/")
    print(f"  │   ├── CT-KIDNEY-DATASET-Normal/")
    print(f"  │   │   ├── Cyst/           (images)")
    print(f"  │   │   ├── Normal/         (images)")
    print(f"  │   │   ├── Stone/          (images)")
    print(f"  │   │   └── Tumor/          (images)")
    print(f"  │   └── kidneyData.csv")
    print(f"  OR place class folders directly in data_dir/")
