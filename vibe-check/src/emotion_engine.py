"""
Emotion Engine - CNN Inference Wrapper.

Loads the trained EmotionCNN model and provides a simple interface
for emotion prediction from face images.

Architecture Decisions:
-----------------------
1. SINGLETON PATTERN: Model loaded once and reused for all predictions.
   Loading the model for each frame would be prohibitively slow.

2. FACE CROPPING: Extract face region using landmarks before prediction.
   This focuses the CNN on the face, improving accuracy.

3. PREPROCESSING: Apply same normalization as training (mean=0.5, std=0.5).
   Consistency between training and inference is critical.

4. SOFTMAX OUTPUT: Return probability distribution over all emotions.
   Enables nuanced analysis (e.g., "60% happy, 30% neutral, 10% surprise").

5. CPU INFERENCE: Real-time webcam can't wait for GPU transfer overhead.
   The small CNN runs fast enough on CPU (~5ms per frame).

Emotion Labels (FER2013):
- 0: Angry   - 1: Disgust   - 2: Fear
- 3: Happy   - 4: Sad       - 5: Surprise
- 6: Neutral
"""
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
import torch
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from models.emotion.model import EmotionCNN, create_emotion_model


class EmotionEngine:
    """
    Emotion prediction engine using trained CNN.
    
    Loads the trained model weights and provides emotion predictions
    from face images. Designed for real-time inference.
    
    Usage:
        engine = EmotionEngine()
        
        # For each frame:
        face = detector.detect(frame)
        if face:
            probs = engine.predict(frame, face)
            print(f"Dominant emotion: {engine.get_dominant_emotion(probs)}")
    
    Args:
        model_path: Path to trained model weights
        device: 'cuda' or 'cpu' (default: auto-detect)
        confidence_threshold: Minimum confidence to consider a prediction valid
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.3
    ):
        self.model_path = model_path or settings.model_path
        self.confidence_threshold = confidence_threshold
        
        # Determine device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = None
        self.emotion_labels = list(settings.emotion_labels)
        self._load_model()
        
        # Normalization parameters (same as training)
        self.normalize_mean = 0.5
        self.normalize_std = 0.5
        
        logger.info(f"EmotionEngine initialized on {self.device}")
    
    def _load_model(self):
        """Load the trained model weights."""
        if not self.model_path.exists():
            logger.warning(f"Model weights not found at {self.model_path}")
            logger.warning("Train the model first: python -m models.emotion.train")
            self.model = None
            return
        
        try:
            # Create model
            self.model = EmotionCNN(
                num_classes=settings.num_emotions,
                dropout_rate=0.0  # No dropout during inference
            )
            
            # Load weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Set to evaluation mode
            self.model.eval()
            self.model.to(self.device)
            
            logger.success(f"Loaded model weights from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self.model is not None
    
    def preprocess_face(
        self,
        frame: np.ndarray,
        face_data
    ) -> Optional[torch.Tensor]:
        """
        Preprocess face image for the CNN.
        
        Steps:
        1. Extract face region using bounding box
        2. Convert to grayscale
        3. Resize to 48x48 (FER2013 native resolution)
        4. Normalize to [-1, 1]
        5. Convert to tensor [1, 1, 48, 48]
        
        Args:
            frame: BGR image from OpenCV
            face_data: FaceData from FaceDetector
        
        Returns:
            Preprocessed tensor ready for the model, or None if preprocessing fails
        """
        try:
            # Extract face region
            x, y, w, h = face_data.bbox
            
            # Ensure bounds are valid
            h_frame, w_frame = frame.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_frame - x)
            h = min(h, h_frame - y)
            
            if w <= 0 or h <= 0:
                return None
            
            face_img = frame[y:y+h, x:x+w]
            
            # Convert to grayscale
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img
            
            # Resize to 48x48
            resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            # Apply training normalization
            normalized = (normalized - self.normalize_mean) / self.normalize_std
            
            # Convert to tensor [1, 48, 48]
            tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
            
            return tensor.float()
            
        except Exception as e:
            logger.error(f"Face preprocessing failed: {e}")
            return None
    
    def predict(
        self,
        frame: np.ndarray,
        face_data
    ) -> Dict[str, float]:
        """
        Predict emotion probabilities from a face image.
        
        Args:
            frame: BGR image from OpenCV
            face_data: FaceData from FaceDetector
        
        Returns:
            Dictionary mapping emotion names to probabilities
        """
        if not self.is_ready():
            # Return uniform distribution if model not loaded
            return {label: 1.0/7 for label in self.emotion_labels}
        
        # Preprocess
        tensor = self.preprocess_face(frame, face_data)
        if tensor is None:
            return {label: 1.0/7 for label in self.emotion_labels}
        
        # Move to device
        tensor = tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        
        # Convert to dictionary
        probs_np = probs.cpu().numpy()[0]
        emotion_probs = {
            label: float(prob) 
            for label, prob in zip(self.emotion_labels, probs_np)
        }
        
        return emotion_probs
    
    def get_dominant_emotion(self, probs: Dict[str, float]) -> str:
        """Get the emotion with highest probability."""
        return max(probs.items(), key=lambda x: x[1])[0]
    
    def get_dominant_confidence(self, probs: Dict[str, float]) -> float:
        """Get the confidence of the dominant emotion."""
        return max(probs.values())
    
    def is_confident(self, probs: Dict[str, float]) -> bool:
        """Check if the prediction is confident enough."""
        return self.get_dominant_confidence(probs) >= self.confidence_threshold
    
    def get_top_emotions(
        self, 
        probs: Dict[str, float], 
        n: int = 3
    ) -> list:
        """Get the top N emotions with their probabilities."""
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:n]


# ── Singleton Instance ──────────────────────────────────────────────────────

_engine_instance: Optional[EmotionEngine] = None


def get_emotion_engine() -> EmotionEngine:
    """Get or create the singleton EmotionEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = EmotionEngine()
    return _engine_instance


# ── Testing ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test the emotion engine."""
    import sys
    from pathlib import Path
    
    # Import face detector
    sys.path.insert(0, str(Path(__file__).parent))
    from face_detector import FaceDetector
    
    logger.info("Testing EmotionEngine...")
    
    # Initialize
    detector = FaceDetector()
    engine = EmotionEngine()
    
    if not engine.is_ready():
        logger.error("Model not trained! Run: python -m models.emotion.train")
        detector.close()
        exit()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    logger.info("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        face = detector.detect(frame)
        
        if face:
            # Predict emotion
            probs = engine.predict(frame, face)
            dominant = engine.get_dominant_emotion(probs)
            confidence = engine.get_dominant_confidence(probs)
            
            # Draw mesh
            frame = detector.draw_mesh(frame, face)
            
            # Draw bbox
            x, y, w, h = face.bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 150), 2)
            
            # Display emotion
            text = f"{dominant.upper()}: {confidence*100:.0f}%"
            cv2.putText(frame, text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2)
            
            # Display all probabilities
            y_offset = y + h + 20
            for emotion, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                text = f"{emotion}: {prob*100:.1f}%"
                cv2.putText(frame, text, (x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 18
        
        cv2.imshow("Emotion Engine Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    logger.success("Test complete!")
