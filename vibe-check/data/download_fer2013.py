"""
FER2013 Dataset Downloader.

Downloads and prepares the FER2013 facial expression dataset from Kaggle.

FER2013 Dataset Info:
- Source: ICML 2013 Workshop on Challenges in Representation Learning
- Size: 35,887 grayscale images (48x48 pixels)
- Classes: 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral)
- Split: 28,709 training, 3,589 validation, 3,589 test

Class Distribution (highly imbalanced):
- Happy:     7,216 samples (20.1%)
- Neutral:   4,965 samples (13.8%)
- Sad:       4,830 samples (13.5%)
- Fear:      4,097 samples (11.4%)
- Angry:     3,995 samples (11.1%)
- Surprise:  3,171 samples (8.8%)
- Disgust:     113 samples (0.3%)  ← Severe imbalance!

Architecture decisions:
- Uses Kaggle API for reliable download
- Saves as numpy arrays for fast loading during training
- Creates separate train/val/test splits
- Computes and logs class weights for handling imbalance
"""
import os
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings


# Emotion label mapping (Kaggle format)
EMOTION_LABELS = {
    0: "angry",
    1: "disgust", 
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}


def check_kaggle_credentials() -> bool:
    """
    Verify Kaggle API credentials exist.
    
    Credentials should be at ~/.kaggle/kaggle.json
    Get them from: https://www.kaggle.com/settings/account
    """
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_path.exists():
        logger.error("Kaggle credentials not found!")
        logger.error(f"Expected location: {kaggle_path}")
        logger.error("")
        logger.error("To get Kaggle API credentials:")
        logger.error("1. Go to https://www.kaggle.com/settings/account")
        logger.error("2. Scroll to 'API' section")
        logger.error("3. Click 'Create New API Token'")
        logger.error("4. Move downloaded kaggle.json to ~/.kaggle/")
        logger.error("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Check permissions (should be 600)
    stat_info = kaggle_path.stat()
    if stat_info.st_mode & 0o777 != 0o600:
        logger.warning(f"Setting secure permissions on {kaggle_path}")
        os.chmod(kaggle_path, 0o600)
    
    logger.success(f"Kaggle credentials found at {kaggle_path}")
    return True


def download_from_kaggle(output_dir: Path) -> Path:
    """
    Download FER2013 dataset from Kaggle.
    
    Uses kaggle CLI tool for reliable, resumable downloads.
    Dataset: https://www.kaggle.com/datasets/msambare/fer2013
    """
    import subprocess
    import shutil
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    kaggle_executable = shutil.which("kaggle") or shutil.which("kaggle.exe")
    if not kaggle_executable:
        # Try finding it in user scripts if not in PATH
        potential_scripts = Path.home() / "AppData" / "Local" / "Python" / "pythoncore-3.14-64" / "Scripts" / "kaggle.exe"
        if potential_scripts.exists():
            kaggle_executable = str(potential_scripts)
    
    if not kaggle_executable:
        logger.error("Kaggle CLI not installed or not in PATH!")
        logger.error("Install with: pip install kaggle")
        raise FileNotFoundError("kaggle executable not found")

    logger.info(f"Using kaggle executable: {kaggle_executable}")
    logger.info("Downloading FER2013 from Kaggle...")
    logger.info("Dataset: msambare/fer2013")
    logger.info(f"Output: {output_dir}")
    
    try:
        # Use kaggle API to download
        result = subprocess.run(
            [kaggle_executable, "datasets", "download", 
             "-d", "msambare/fer2013",
             "-p", str(output_dir),
             "--unzip"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Kaggle download failed: {result.stderr}")
            raise RuntimeError("Download failed")
        
        logger.success("Download complete!")
        
        # Find the CSV file
        csv_files = list(output_dir.glob("**/*.csv"))
        if csv_files:
            return csv_files[0]
        
        # Alternative: check for train/test folders
        train_dir = output_dir / "train"
        test_dir = output_dir / "test"
        if train_dir.exists() and test_dir.exists():
            logger.info("Dataset in folder format (train/test)")
            return output_dir
        
        raise FileNotFoundError("Could not find dataset files")
        
    except FileNotFoundError:
        logger.error("Kaggle CLI not installed!")
        logger.error("Install with: pip install kaggle")
        raise


def parse_pixels(pixels_str: str) -> np.ndarray:
    """
    Parse pixel string from CSV into 48x48 numpy array.
    
    FER2013 CSV stores pixels as space-separated integers:
    "23 45 67 89 ..." → np.array([23, 45, 67, 89, ...]).reshape(48, 48)
    """
    pixels = np.array(pixels_str.split(), dtype=np.uint8)
    return pixels.reshape(48, 48)


def load_csv_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load FER2013 from CSV format.
    
    CSV columns: emotion, pixels, Usage
    - emotion: int (0-6, see EMOTION_LABELS)
    - pixels: space-separated uint8 values (48x48 = 2304 values)
    - Usage: "Training", "PublicTest" (val), or "PrivateTest" (test)
    
    Returns: (train_data, val_data, test_data)
    Each: dict with 'images' and 'labels' keys
    """
    logger.info(f"Loading dataset from {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Total samples: {len(df)}")
    
    # Parse all images
    logger.info("Parsing pixel data...")
    images = np.stack(df['pixels'].apply(parse_pixels).values)
    images = images.reshape(-1, 48, 48, 1)  # Add channel dimension
    labels = df['emotion'].values
    
    # Split by Usage column
    train_mask = df['Usage'] == 'Training'
    val_mask = df['Usage'] == 'PublicTest'
    test_mask = df['Usage'] == 'PrivateTest'
    
    train_data = {
        'images': images[train_mask],
        'labels': labels[train_mask]
    }
    val_data = {
        'images': images[val_mask],
        'labels': labels[val_mask]
    }
    test_data = {
        'images': images[test_mask],
        'labels': labels[test_mask]
    }
    
    logger.success(f"Train: {len(train_data['labels'])} samples")
    logger.success(f"Val:   {len(val_data['labels'])} samples")
    logger.success(f"Test:  {len(test_data['labels'])} samples")
    
    return train_data, val_data, test_data


def load_folder_dataset(data_dir: Path) -> Tuple[dict, dict, dict]:
    """
    Load FER2013 from folder format (train/test subdirectories).
    
    Structure:
    train/
      angry/
      disgust/
      ...
    test/
      angry/
      disgust/
      ...
    """
    logger.info(f"Loading dataset from folder: {data_dir}")
    
    def load_split(split_name: str) -> dict:
        split_dir = data_dir / split_name
        images = []
        labels = []
        
        for label_idx, emotion in EMOTION_LABELS.items():
            emotion_dir = split_dir / emotion
            if not emotion_dir.exists():
                continue
            
            for img_path in emotion_dir.glob("*.jpg"):
                import cv2
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img.reshape(48, 48, 1))
                    labels.append(label_idx)
        
        return {
            'images': np.array(images),
            'labels': np.array(labels)
        }
    
    train_data = load_split('train')
    test_data = load_split('test')
    
    # Split train into train/val (90/10)
    n_train = len(train_data['labels'])
    n_val = int(n_train * 0.1)
    
    indices = np.random.permutation(n_train)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    val_data = {
        'images': train_data['images'][val_indices],
        'labels': train_data['labels'][val_indices]
    }
    train_data = {
        'images': train_data['images'][train_indices],
        'labels': train_data['labels'][train_indices]
    }
    
    logger.success(f"Train: {len(train_data['labels'])} samples")
    logger.success(f"Val:   {len(val_data['labels'])} samples")
    logger.success(f"Test:  {len(test_data['labels'])} samples")
    
    return train_data, val_data, test_data


def compute_class_weights(labels: np.ndarray) -> dict:
    """
    Compute class weights to handle dataset imbalance.
    
    Disgust has only 113 samples while Happy has 7,216!
    Using class weights ensures the model doesn't ignore minority classes.
    
    Formula: weight = n_samples / (n_classes * class_count)
    """
    from collections import Counter
    
    counts = Counter(labels)
    n_classes = len(EMOTION_LABELS)
    n_samples = len(labels)
    
    weights = {}
    for class_idx in range(n_classes):
        class_count = counts.get(class_idx, 1)
        # Balanced class weight formula
        weights[class_idx] = n_samples / (n_classes * class_count)
    
    logger.info("Class weights (for handling imbalance):")
    for idx, weight in weights.items():
        emotion = EMOTION_LABELS[idx]
        count = counts.get(idx, 0)
        logger.info(f"  {emotion:10s}: weight={weight:.2f} (n={count})")
    
    return weights


def compute_normalization_stats(images: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean and std for image normalization.
    
    Pre-computing these values allows for consistent normalization
    during training and inference.
    """
    mean = images.mean() / 255.0
    std = images.std() / 255.0
    
    logger.info(f"Normalization stats:")
    logger.info(f"  Mean: {mean:.4f}")
    logger.info(f"  Std:  {std:.4f}")
    
    return mean, std


def save_dataset(train_data: dict, val_data: dict, test_data: dict, 
                 output_dir: Path):
    """
    Save processed dataset as numpy arrays.
    
    Using .npz format for efficient storage and fast loading.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each split
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_path = output_dir / f"fer2013_{name}.npz"
        np.savez_compressed(
            output_path,
            images=data['images'],
            labels=data['labels']
        )
        logger.success(f"Saved {output_path}")
    
    # Save metadata
    class_weights = compute_class_weights(train_data['labels'])
    mean, std = compute_normalization_stats(train_data['images'])
    
    metadata_path = output_dir / "fer2013_metadata.npz"
    np.savez(
        metadata_path,
        class_weights=np.array([class_weights[i] for i in range(7)]),
        mean=mean,
        std=std,
        emotion_labels=np.array(list(EMOTION_LABELS.values()))
    )
    logger.success(f"Saved metadata to {metadata_path}")


def main():
    """Main download and processing pipeline."""
    logger.info("=" * 60)
    logger.info("FER2013 Dataset Downloader")
    logger.info("=" * 60)
    
    # Check credentials
    if not check_kaggle_credentials():
        sys.exit(1)
    
    # Setup paths
    output_dir = settings.data_path
    processed_dir = output_dir / "processed"
    
    # Download
    try:
        dataset_path = download_from_kaggle(output_dir)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)
    
    # Load dataset
    logger.info("")
    logger.info("Processing dataset...")
    
    if dataset_path.suffix == '.csv':
        train_data, val_data, test_data = load_csv_dataset(dataset_path)
    else:
        train_data, val_data, test_data = load_folder_dataset(dataset_path)
    
    # Save processed data
    logger.info("")
    logger.info("Saving processed dataset...")
    save_dataset(train_data, val_data, test_data, processed_dir)
    
    # Print summary
    logger.info("")
    logger.success("=" * 60)
    logger.success("FER2013 Dataset Ready!")
    logger.success("=" * 60)
    logger.success(f"Processed files saved to: {processed_dir}")
    logger.success("")
    logger.success("Next step: Train the model")
    logger.success("  python -m models.emotion.train")
    
    # Print class distribution warning
    logger.warning("")
    logger.warning("Note: FER2013 is highly imbalanced!")
    logger.warning("'Disgust' has only 113 samples (0.3%)")
    logger.warning("Class weights will be used during training")


if __name__ == "__main__":
    main()
