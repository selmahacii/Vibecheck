"""
Configuration settings for Vibe Check.
Uses pydantic-settings for type-safe, environment-variable-friendly config.
All parameters are centralized here for easy tuning.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from pathlib import Path


class Settings(BaseSettings):
    """
    Global configuration for the Vibe Check emotion detection system.
    
    Architecture decisions:
    - pydantic-settings: Allows env var overrides (useful for Docker/deployment)
    - Path(__file__).parent: All paths relative to this config file
    - Small CNN: FER2013 images are only 48x48, deep networks overfit quickly
    """
    
    # ── Camera Settings ─────────────────────────────────────────────────────
    camera_index: int = Field(
        default=0,
        description="OpenCV camera index (0=default, 1=external)"
    )
    frame_width: int = Field(
        default=640,
        description="Camera resolution width"
    )
    frame_height: int = Field(
        default=480,
        description="Camera resolution height"
    )
    target_fps: int = Field(
        default=30,
        description="Target frames per second for camera capture"
    )
    
    # ── Model Settings ──────────────────────────────────────────────────────
    # FER2013 has 35,887 grayscale images, 48x48 pixels
    # 7 emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
    num_emotions: int = 7
    emotion_labels: tuple[str, ...] = (
        "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
    )
    input_size: int = 48  # FER2013 native resolution
    
    # ── Training Hyperparameters ────────────────────────────────────────────
    batch_size: int = Field(
        default=64,
        description="Batch size for training (larger = faster but more VRAM)"
    )
    learning_rate: float = Field(
        default=0.001,
        description="Initial learning rate for Adam optimizer"
    )
    num_epochs: int = Field(
        default=50,
        description="Number of training epochs"
    )
    weight_decay: float = Field(
        default=1e-4,
        description="L2 regularization strength"
    )
    dropout_rate: float = Field(
        default=0.5,
        description="Dropout probability for regularization"
    )
    
    # ── Face Detection Settings ─────────────────────────────────────────────
    # MediaPipe FaceMesh: 468 landmarks per face
    # Static mode=False for video (uses temporal smoothing)
    face_mesh_static_mode: bool = False
    face_mesh_max_faces: int = 1  # Single person dashboard
    face_mesh_refine_landmarks: bool = True  # Adds iris landmarks
    
    # ── Eye Aspect Ratio (EAR) for Attention ─────────────────────────────────
    # EAR threshold for blink detection (lower = more closed eyes)
    ear_blink_threshold: float = 0.2
    # Time window for blink rate calculation (seconds)
    blink_window_seconds: float = 60.0
    
    # ── History Buffer Settings ─────────────────────────────────────────────
    history_seconds: int = 60  # Store 60 seconds of timeline data
    history_fps: int = 10  # Sample rate for history (downsampled from 30fps)
    
    # ── Paths ───────────────────────────────────────────────────────────────
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Base directory of the project"
    )
    
    @property
    def weights_path(self) -> Path:
        """Directory for trained model weights."""
        return self.base_dir / "weights"
    
    @property
    def data_path(self) -> Path:
        """Directory for datasets."""
        return self.base_dir / "data"
    
    @property
    def model_path(self) -> Path:
        """Path to the trained emotion model."""
        return self.weights_path / "emotion_cnn.pth"
    
    # ── API Server Settings ─────────────────────────────────────────────────
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    api_port: int = Field(
        default=5000,
        description="API server port"
    )
    
    # ── Logging ─────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    class Config:
        env_prefix = "VIBE_"  # Environment variables: VIBE_CAMERA_INDEX=1
        env_file = ".env"


# Global singleton instance
settings = Settings()
