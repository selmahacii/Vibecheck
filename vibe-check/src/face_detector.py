"""
Face Detector using MediaPipe FaceMesh (v0.10+ Tasks API).

Provides real-time face detection with 478 landmark points (including iris).
These landmarks enable:
- Precise face bounding box
- Eye Aspect Ratio (EAR) for blink detection
- Gaze direction estimation
- Face mesh visualization

Architecture Decisions:
-----------------------
1. MEDIAPIPE TASKS API: Replaces the legacy solutions API which is missing 
   in newer Python environments (like Python 3.14).
2. REFINE_LANDMARKS: Handled by the model itself (478 points).
3. MODEL_PATH: Requires a .task file in weights directory.

MediaPipe FaceMesh Landmark Indices (consistent with old API):
------------------------------------------------------------
- Left eye: 33, 133, 159, 145, 153, 154, 155, 157, 158, 160, 161, 163, 173
- Right eye: 362, 263, 386, 373, 382, 381, 380, 390, 387, 388, 384, 398, 368
- Nose tip: 1
- Chin: 152
- Forehead: 10
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
from typing import Optional, List, Tuple
from loguru import logger
from pathlib import Path


@dataclass
class FaceData:
    """Container for face detection results (Indices compatible with EyeAnalyzer)."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    landmarks: np.ndarray  # Shape: (N, 3) - normalized
    landmarks_px: np.ndarray  # Shape: (N, 2) - pixel coords
    left_eye_indices: List[int]
    right_eye_indices: List[int]
    left_iris_center: Optional[Tuple[int, int]]
    right_iris_center: Optional[Tuple[int, int]]
    confidence: float
    
    # Key landmark indices for quick access
    NOSE_TIP = 1
    CHIN = 152
    FOREHEAD = 10
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_INNER = 362
    
    # Eye landmark indices for EAR calculation (6 points per eye)
    LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]


class FaceDetector:
    """
    Detector using the modern FaceLandmarker (Tasks API).
    Expected landmarks: 478 points.
    """
    
    def __init__(
        self,
        static_mode: bool = False,
        max_faces: int = 1,
        min_confidence: float = 0.5,
        model_path: str = "weights/face_landmarker.task"
    ):
        # Resolve path
        base_dir = Path(__file__).parent.parent
        abs_model_path = base_dir / model_path
        
        if not abs_model_path.exists():
            raise FileNotFoundError(f"FaceLandmarker model not found at {abs_model_path}")

        # Setup FaceLandmarker options
        base_options = python.BaseOptions(model_asset_path=str(abs_model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        
        # Initialize detector
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.refine_landmarks = True # Tasks API defaults to 478 landmarks (full mesh)
        
        # Indices for EAR calculation (same as legacy)
        self.left_eye_ear_idx = np.array([33, 160, 158, 133, 153, 144])
        self.right_eye_ear_idx = np.array([362, 385, 387, 263, 373, 380])
        
        # Iris indices (landmarks 468-477)
        self.left_iris_idx = list(range(468, 473)) # 5 points per iris in new model
        self.right_iris_idx = list(range(473, 478))
        
        logger.info(f"FaceDetector (Tasks API) initialized using {model_path}")
    
    def detect(self, frame: np.ndarray) -> Optional[FaceData]:
        """Convert BGR frame to MP Image and process."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Wrap in Mediapipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process detection
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None
            
        # Extract first face landmarks
        face_landmarks = detection_result.face_landmarks[0]
        h, w = frame.shape[:2]
        
        landmarks = []
        landmarks_px = []
        for lm in face_landmarks:
            landmarks.append([lm.x, lm.y, lm.z])
            landmarks_px.append([int(lm.x * w), int(lm.y * h)])
            
        landmarks = np.array(landmarks)
        landmarks_px = np.array(landmarks_px)
        
        # Bounding Box
        x_min, y_min = landmarks_px.min(axis=0)
        x_max, y_max = landmarks_px.max(axis=0)
        
        # Padding
        pw = int((x_max - x_min) * 0.1)
        ph = int((y_max - y_min) * 0.1)
        x = max(0, int(x_min - pw))
        y = max(0, int(y_min - ph))
        width = min(w - x, int(x_max - x_min + 2 * pw))
        height = min(h - y, int(y_max - y_min + 2 * ph))
        
        # Iris
        l_iris_center = None
        r_iris_center = None
        if len(landmarks_px) >= 478:
            l_iris_px = landmarks_px[self.left_iris_idx]
            r_iris_px = landmarks_px[self.right_iris_idx]
            l_iris_center = tuple(l_iris_px.mean(axis=0).astype(int))
            r_iris_center = tuple(r_iris_px.mean(axis=0).astype(int))
            
        return FaceData(
            bbox=(x, y, width, height),
            landmarks=landmarks,
            landmarks_px=landmarks_px,
            left_eye_indices=self.left_eye_ear_idx.tolist(),
            right_eye_indices=self.right_eye_ear_idx.tolist(),
            left_iris_center=l_iris_center,
            right_iris_center=r_iris_center,
            confidence=1.0
        )
        
    def draw_mesh(self, frame: np.ndarray, face: FaceData, 
                  color: Tuple[int,int,int]=(0,255,150), thickness: int=1) -> np.ndarray:
        output = frame.copy()
        for x, y in face.landmarks_px:
            cv2.circle(output, (int(x), int(y)), 1, color, -1)
        if face.left_iris_center:
            cv2.circle(output, face.left_iris_center, 3, (0, 255, 255), -1)
            cv2.circle(output, face.right_iris_center, 3, (0, 255, 255), -1)
        return output
        
    def close(self):
        self.detector.close()
        logger.info("FaceDetector closed")

def compute_ear(landmarks_px: np.ndarray, eye_indices: List[int]) -> float:
    eye = landmarks_px[eye_indices].astype(np.float64)
    p1, p2, p3, p4, p5, p6 = eye
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    if h == 0: return 0.0
    return (v1 + v2) / (2.0 * h)

def compute_gaze_direction(l_iris, r_iris, l_center, r_center) -> Tuple[float, float]:
    if l_iris is None or r_iris is None: return (0.0, 0.0)
    gx = ((l_iris[0]-l_center[0])/30.0 + (r_iris[0]-r_center[0])/30.0)/2.0
    gy = ((l_iris[1]-l_center[1])/15.0 + (r_iris[1]-r_center[1])/15.0)/2.0
    return (float(np.clip(gx, -1, 1)), float(np.clip(gy, -1, 1)))
