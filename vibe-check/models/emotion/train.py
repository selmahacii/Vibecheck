"""
Training Script for FER2013 Emotion Classification.

Trains the EmotionCNN model on the FER2013 dataset with:
- Data augmentation for better generalization
- Class weights to handle imbalanced data
- Learning rate scheduling
- Early stopping to prevent overfitting
- Model checkpointing

Training Strategy:
------------------
1. AUGMENTATION: Horizontal flip, rotation, and slight scaling.
   FER2013 is small (35K images), augmentation is essential.

2. CLASS WEIGHTS: Disgust has only 113 samples vs Happy's 7,216.
   Without weights, the model would ignore minority classes.

3. LEARNING RATE: Adam optimizer with OneCycleLR schedule.
   - Starts low, peaks mid-training, decreases to end
   - Better than fixed LR or simple decay

4. EARLY STOPPING: Stop if val loss doesn't improve for 10 epochs.
   Prevents overfitting on small dataset.

5. LABEL SMOOTHING: Use 0.9 instead of 1.0 for target labels.
   Prevents overconfident predictions, improves calibration.

Usage:
    python -m models.emotion.train

    Optional arguments:
    --epochs 50          # Number of training epochs
    --batch-size 64      # Batch size
    --lr 0.001           # Learning rate
    --no-augment         # Disable augmentation
    --cpu                # Force CPU training
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import settings
from models.emotion.model import EmotionCNN, create_emotion_model, count_parameters


# ── Custom Dataset ──────────────────────────────────────────────────────────

class FER2013Dataset(Dataset):
    """
    PyTorch Dataset for FER2013.
    
    Loads preprocessed numpy arrays and applies augmentation during training.
    
    Args:
        npz_path: Path to .npz file with 'images' and 'labels' arrays
        augment: Whether to apply data augmentation
        normalize_mean: Mean for normalization (default 0.5)
        normalize_std: Std for normalization (default 0.5)
    """
    
    def __init__(
        self, 
        npz_path: Path,
        augment: bool = False,
        normalize_mean: float = 0.5,
        normalize_std: float = 0.5
    ):
        self.augment = augment
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        # Load data
        logger.info(f"Loading dataset from {npz_path}")
        data = np.load(npz_path)
        self.images = data['images']
        self.labels = data['labels']
        
        logger.info(f"Loaded {len(self.labels)} samples")
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get image and label
        image = self.images[idx].astype(np.float32)
        label = int(self.labels[idx])
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Apply augmentation
        if self.augment:
            image = self._augment(image)
        
        # Normalize to [-1, 1] range (mean=0.5, std=0.5)
        image = (image - self.normalize_mean) / self.normalize_std
        
        # Convert to tensor [1, 48, 48]
        if len(image.shape) == 2:
            image = image.reshape(48, 48, 1)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image, label
    
    def _augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation.
        
        Augmentations used:
        - Horizontal flip (50%): Faces are horizontally symmetric
        - Rotation (±10°): Head tilt is common
        - Shift (±5%): Face position varies
        - Random brightness/contrast: Lighting changes
        
        NOT used:
        - Vertical flip: Upside-down faces are not realistic
        - Large rotations: Would distort facial features
        """
        # Horizontal flip (50% chance)
        if np.random.random() < 0.5:
            image = np.fliplr(image).copy()
        
        # Rotation (±10 degrees)
        angle = np.random.uniform(-10, 10)
        image = self._rotate_image(image, angle)
        
        # Small shift (±5% of image size)
        shift_x = np.random.randint(-2, 3)
        shift_y = np.random.randint(-2, 3)
        image = np.roll(image, shift_x, axis=1)
        image = np.roll(image, shift_y, axis=0)
        
        # Brightness adjustment (±10%)
        brightness = np.random.uniform(0.9, 1.1)
        image = np.clip(image * brightness, 0, 1)
        
        # Contrast adjustment (±10%)
        contrast = np.random.uniform(0.9, 1.1)
        mean = image.mean()
        image = np.clip((image - mean) * contrast + mean, 0, 1)
        
        return image.astype(np.float32)
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle using scipy or opencv."""
        try:
            import cv2
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, matrix, (w, h), 
                                      borderMode=cv2.BORDER_REPLICATE)
            return rotated.astype(np.float32)
        except ImportError:
            # Fallback: no rotation if cv2 not available
            return image


# ── Training Functions ──────────────────────────────────────────────────────

def get_class_weights(data_path: Path) -> torch.Tensor:
    """
    Load or compute class weights for imbalanced dataset.
    
    Uses the inverse frequency weighting scheme:
        weight[i] = n_total / (n_classes * n_samples[i])
    """
    # Check for pre-computed weights
    metadata_path = data_path / "fer2013_metadata.npz"
    if metadata_path.exists():
        metadata = np.load(metadata_path)
        weights = metadata['class_weights']
        logger.info("Loaded pre-computed class weights")
        return torch.tensor(weights, dtype=torch.float32)
    
    # Compute from training data
    train_data = np.load(data_path / "fer2013_train.npz")
    labels = train_data['labels']
    
    n_samples = len(labels)
    n_classes = 7
    
    # Count samples per class
    counts = np.bincount(labels, minlength=n_classes)
    
    # Compute weights
    weights = n_samples / (n_classes * counts)
    weights = weights / weights.sum() * n_classes  # Normalize
    
    logger.info("Computed class weights from training data:")
    for i, (count, weight) in enumerate(zip(counts, weights)):
        logger.info(f"  Class {i}: {count} samples, weight={weight:.3f}")
    
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        avg_loss: Average training loss
        accuracy: Training accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update scheduler (for OneCycleLR)
        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # Statistics
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.*correct/total:.1f}%"
        })
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray]:
    """
    Validate the model.
    
    Returns:
        avg_loss: Average validation loss
        accuracy: Validation accuracy
        confusion_matrix: Per-class accuracy breakdown
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class statistics
    class_correct = np.zeros(7)
    class_total = np.zeros(7)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Per-class accuracy
            for label, pred in zip(labels, predicted):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    # Class accuracies
    class_accuracy = class_correct / np.maximum(class_total, 1)
    
    return avg_loss, accuracy, class_accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    args: argparse.Namespace
) -> nn.Module:
    """
    Full training loop.
    
    Implements:
    - Adam optimizer with weight decay
    - OneCycleLR scheduler
    - Early stopping
    - Model checkpointing
    - Label smoothing
    """
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Training on device: {device}")
    
    model = model.to(device)
    class_weights = class_weights.to(device)
    
    # Loss function with label smoothing
    # Label smoothing: target = 1-ε for correct class, ε/(C-1) for others
    # Helps prevent overconfident predictions
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1  # Smooth labels: 0.9 for correct, 0.1/6 for others
    )
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler: OneCycleLR
    # - Gradually increases LR from 0 to max
    # - Then decreases back to near 0
    # - Better than fixed LR or simple decay
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 10,  # Peak learning rate
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Spend 30% of training increasing LR
        anneal_strategy='cos',
        div_factor=25,  # Initial LR = max_lr/25
        final_div_factor=1000  # Final LR = max_lr/1000
    )
    
    # Training tracking
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Augmentation: {args.augment}")
    logger.info("=" * 60)
    
    emotion_labels = settings.emotion_labels
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        
        # Validate
        val_loss, val_acc, class_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Print per-class accuracy every 5 epochs
        if epoch % 5 == 0:
            logger.info("  Per-class accuracy:")
            for i, (label, acc) in enumerate(zip(emotion_labels, class_acc)):
                logger.info(f"    {label:10s}: {acc*100:.1f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            save_path = settings.weights_path / "emotion_cnn_best.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'class_accuracy': class_acc,
            }, save_path)
            logger.success(f"  Saved best model (val_loss={val_loss:.4f})")
        
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.warning(f"Early stopping after {epoch} epochs")
                break
    
    # Load best model
    best_path = settings.weights_path / "emotion_cnn_best.pth"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.success(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    # Save final model
    final_path = settings.weights_path / "emotion_cnn.pth"
    torch.save(model.state_dict(), final_path)
    logger.success(f"Saved final model to {final_path}")
    
    # Save training history
    history_path = settings.weights_path / "training_history.npz"
    np.savez(history_path, **history)
    logger.success(f"Saved training history to {history_path}")
    
    logger.info("=" * 60)
    logger.success("Training Complete!")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    logger.info(f"Best Validation Accuracy: {best_val_acc*100:.1f}%")
    logger.info("=" * 60)
    
    return model


def test(model: nn.Module, test_loader: DataLoader, device: torch.device):
    """Evaluate on test set and print detailed metrics."""
    logger.info("\nEvaluating on test set...")
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_loss, test_acc, class_acc = validate(model, test_loader, criterion, device)
    
    emotion_labels = settings.emotion_labels
    
    logger.info("=" * 60)
    logger.info("Test Set Results")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy: {test_acc*100:.2f}%")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info("")
    logger.info("Per-Class Accuracy:")
    for label, acc in zip(emotion_labels, class_acc):
        logger.info(f"  {label:10s}: {acc*100:.1f}%")
    logger.info("=" * 60)
    
    return test_acc


# ── Main Entry Point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train EmotionCNN on FER2013"
    )
    parser.add_argument(
        '--epochs', type=int, default=settings.num_epochs,
        help=f'Number of training epochs (default: {settings.num_epochs})'
    )
    parser.add_argument(
        '--batch-size', type=int, default=settings.batch_size,
        help=f'Batch size (default: {settings.batch_size})'
    )
    parser.add_argument(
        '--lr', type=float, default=settings.learning_rate,
        help=f'Learning rate (default: {settings.learning_rate})'
    )
    parser.add_argument(
        '--weight-decay', type=float, default=settings.weight_decay,
        help=f'Weight decay for L2 regularization (default: {settings.weight_decay})'
    )
    parser.add_argument(
        '--patience', type=int, default=10,
        help='Early stopping patience (default: 10)'
    )
    parser.add_argument(
        '--augment', action='store_true', default=True,
        help='Use data augmentation (default: True)'
    )
    parser.add_argument(
        '--no-augment', action='store_false', dest='augment',
        help='Disable data augmentation'
    )
    parser.add_argument(
        '--cpu', action='store_true',
        help='Force CPU training (disable CUDA)'
    )
    
    args = parser.parse_args()
    
    # Check for dataset
    data_path = settings.data_path / "processed"
    train_path = data_path / "fer2013_train.npz"
    val_path = data_path / "fer2013_val.npz"
    test_path = data_path / "fer2013_test.npz"
    
    if not train_path.exists():
        logger.error("Dataset not found!")
        logger.error("Please download and process the dataset first:")
        logger.error("  python -m data.download_fer2013")
        sys.exit(1)
    
    # Create datasets
    train_dataset = FER2013Dataset(train_path, augment=args.augment)
    val_dataset = FER2013Dataset(val_path, augment=False)
    test_dataset = FER2013Dataset(test_path, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Get class weights
    class_weights = get_class_weights(data_path)
    
    # Create model
    model = create_emotion_model(
        num_classes=settings.num_emotions,
        dropout_rate=settings.dropout_rate
    )
    
    # Train
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = train(model, train_loader, val_loader, class_weights, args)
    
    # Test
    test_acc = test(model, test_loader, device)
    
    logger.success(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
    logger.info("\nTo use the trained model, run the dashboard:")
    logger.info("  streamlit run app/main.py")


if __name__ == "__main__":
    main()
