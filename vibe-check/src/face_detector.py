"""
Face Detector using MediaPipe FaceMesh.

Provides real-time face detection with 468 landmark points.
These landmarks enable:
- Precise face bounding box
- Eye Aspect Ratio (EAR) for blink detection
- Gaze direction estimation
- Face mesh visualization

Architecture Decisions:
-----------------------
1. MEDIAPIPE FACEMESH: Chosen over OpenCV Haar cascades or DNN face detector
   because:
   - Provides 468 landmarks vs just bounding box
   - More accurate and robust to pose/illumination
   - Includes iris landmarks for gaze tracking
   - Runs efficiently on CPU (~30fps)

2. STATIC_MODE=False: For video streams, temporal smoothing improves accuracy.
   Set to True only for single-image analysis.

3. REFINE_LANDMARKS=True: Adds 10 iris landmarks per eye for gaze tracking.
   Essential for attention detection.

4. DATACLASS: FaceData stores all detection results in a clean structure.

MediaPipe FaceMesh Landmark Indices:
-----------------------------------
- Left eye: 33, 133, 159, 145, 153, 154, 155, 157, 158, 160, 161, 163, 173
- Right eye: 362, 263, 386, 373, 382, 381, 380, 390, 387, 388, 384, 398, 368
- Left iris: 468-477 (10 points)
- Right iris: 478-487 (10 points)
- Nose tip: 1
- Chin: 152
- Forehead: 10

Performance:
- CPU: ~30fps at 640x480
- GPU: Not required (MediaPipe is CPU-optimized)
- Memory: ~50MB for model weights
"""
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional, List, Tuple
from loguru import logger


@dataclass
class FaceData:
    """
    Container for face detection results.
    
    Attributes:
        bbox: Bounding box (x, y, width, height)
        landmarks: Normalized landmarks (x, y, z) in [0, 1] range
        landmarks_px: Pixel coordinates of landmarks
        left_eye_indices: Indices for left eye landmarks
        right_eye_indices: Indices for right eye landmarks
        left_iris_center: Center of left iris in pixels
        right_iris_center: Center of right iris in pixels
        confidence: Detection confidence (0-1)
    """
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    landmarks: np.ndarray  # Shape: (468, 3) - normalized
    landmarks_px: np.ndarray  # Shape: (468, 2) - pixel coords
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
    
    # Eye landmark indices for EAR calculation
    # These are the 6 key points for each eye
    LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]  # outer, top, bottom, inner, bottom, top
    RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
    
    def get_eye_landmarks(self, eye: str = 'left') -> np.ndarray:
        """Get pixel coordinates for eye landmarks."""
        if eye == 'left':
            return self.landmarks_px[self.LEFT_EYE_EAR]
        return self.landmarks_px[self.RIGHT_EYE_EAR]


class FaceDetector:
    """
    Face detector using MediaPipe FaceMesh.
    
    Provides 468 facial landmarks for emotion analysis,
    eye tracking, and attention detection.
    
    Usage:
        detector = FaceDetector()
        frame = cv2.imread('face.jpg')
        face = detector.detect(frame)
        if face:
            print(f"Face bbox: {face.bbox}")
            print(f"Landmarks: {len(face.landmarks)}")
    
    Args:
        static_mode: If True, treats each frame independently (slower but more accurate)
        max_faces: Maximum number of faces to detect
        refine_landmarks: If True, adds iris landmarks for gaze tracking
        min_confidence: Minimum confidence for detection
    """
    
    def __init__(
        self,
        static_mode: bool = False,
        max_faces: int = 1,
        refine_landmarks: bool = True,
        min_confidence: float = 0.5
    ):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks
        self.min_confidence = min_confidence
        
        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5
        )
        
        # Pre-compute eye landmark indices for EAR calculation
        # These are the 6 key points forming the eye shape
        self.left_eye_ear_idx = np.array([33, 160, 158, 133, 153, 144])
        self.right_eye_ear_idx = np.array([362, 385, 387, 263, 373, 380])
        
        # Iris indices (if refine_landmarks=True)
        self.left_iris_idx = list(range(468, 478))  # 10 iris landmarks
        self.right_iris_idx = list(range(478, 488))
        
        logger.info(f"FaceDetector initialized (refine_landmarks={refine_landmarks})")
    
    def detect(self, frame: np.ndarray) -> Optional[FaceData]:
        """
        Detect face and extract landmarks from a frame.
        
        Args:
            frame: BGR image from OpenCV (H, W, 3)
        
        Returns:
            FaceData if face detected, None otherwise
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.mp_face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face (we set max_faces=1)
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract landmarks
        h, w = frame.shape[:2]
        landmarks = []
        landmarks_px = []
        
        for landmark in face_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
            landmarks_px.append([int(landmark.x * w), int(landmark.y * h)])
        
        landmarks = np.array(landmarks)
        landmarks_px = np.array(landmarks_px)
        
        # Compute bounding box from landmarks
        x_min = int(landmarks_px[:, 0].min())
        x_max = int(landmarks_px[:, 0].max())
        y_min = int(landmarks_px[:, 1].min())
        y_max = int(landmarks_px[:, 1].max())
        
        # Add padding (10%)
        padding_x = int((x_max - x_min) * 0.1)
        padding_y = int((y_max - y_min) * 0.1)
        
        x = max(0, x_min - padding_x)
        y = max(0, y_min - padding_y)
        width = min(w - x, x_max - x_min + 2 * padding_x)
        height = min(h - y, y_max - y_min + 2 * padding_y)
        
        # Extract iris centers (if available)
        left_iris_center = None
        right_iris_center = None
        
        if self.refine_landmarks and len(landmarks_px) >= 488:
            # Compute iris center as mean of iris landmarks
            left_iris_px = landmarks_px[self.left_iris_idx]
            right_iris_px = landmarks_px[self.right_iris_idx]
            
            left_iris_center = tuple(left_iris_px.mean(axis=0).astype(int))
            right_iris_center = tuple(right_iris_px.mean(axis=0).astype(int))
        
        # Create FaceData
        face_data = FaceData(
            bbox=(x, y, width, height),
            landmarks=landmarks,
            landmarks_px=landmarks_px,
            left_eye_indices=self.left_eye_ear_idx.tolist(),
            right_eye_indices=self.right_eye_ear_idx.tolist(),
            left_iris_center=left_iris_center,
            right_iris_center=right_iris_center,
            confidence=1.0  # MediaPipe doesn't provide per-face confidence
        )
        
        return face_data
    
    def draw_mesh(
        self,
        frame: np.ndarray,
        face: FaceData,
        color: Tuple[int, int, int] = (0, 255, 150),
        thickness: int = 1,
        draw_iris: bool = True
    ) -> np.ndarray:
        """
        Draw face mesh landmarks on the frame.
        
        Draws:
        - All 468 landmark points (small dots)
        - Eye contours (larger dots)
        - Iris centers (if available)
        
        Args:
            frame: BGR image to draw on
            face: FaceData from detect()
            color: RGB color for drawing
            thickness: Line thickness
            draw_iris: Whether to draw iris centers
        
        Returns:
            Frame with mesh drawn
        """
        output = frame.copy()
        
        # Draw all landmarks as small dots
        for x, y in face.landmarks_px:
            cv2.circle(output, (int(x), int(y)), 1, color, -1)
        
        # Draw eye landmarks larger
        for idx in face.left_eye_indices + face.right_eye_indices:
            x, y = face.landmarks_px[idx]
            cv2.circle(output, (int(x), int(y)), 3, color, -1)
        
        # Draw iris centers
        if draw_iris and face.left_iris_center:
            cv2.circle(output, face.left_iris_center, 5, (0, 255, 255), -1)
            cv2.circle(output, face.right_iris_center, 5, (0, 255, 255), -1)
        
        return output
    
    def draw_contour(
        self,
        frame: np.ndarray,
        face: FaceData,
        color: Tuple[int, int, int] = (0, 255, 150),
        thickness: int = 1
    ) -> np.ndarray:
        """
        Draw face contour (oval) on the frame.
        
        Uses key landmarks around the face perimeter to draw
        a smooth contour outline.
        
        Args:
            frame: BGR image to draw on
            face: FaceData from detect()
            color: RGB color for drawing
            thickness: Line thickness
        
        Returns:
            Frame with contour drawn
        """
        output = frame.copy()
        
        # Face contour indices (from MediaPipe documentation)
        # These form the outline of the face
        contour_indices = [
            10, 338, 297, 332, 284, 251, 389, 356,  # Right side
            454, 323, 361, 288, 397, 365, 379, 378,  # Right jaw
            400, 377, 152, 148, 176, 149, 150, 136,  # Chin
            172, 58, 132, 93, 234, 127, 162, 21,     # Left jaw
            54, 103, 67, 109, 10                      # Left side to forehead
        ]
        
        # Draw contour
        points = face.landmarks_px[contour_indices].astype(np.int32)
        cv2.polylines(output, [points], False, color, thickness)
        
        return output
    
    def close(self):
        """Release MediaPipe resources."""
        self.mp_face_mesh.close()
        logger.info("FaceDetector closed")


# ── Utility Functions ────────────────────────────────────────────────────────

def compute_ear(landmarks_px: np.ndarray, eye_indices: List[int]) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for blink detection.
    
    EAR measures eye openness. Lower values indicate closed eyes.
    
    Formula:
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        
    Where p1-p6 are 6 eye landmarks:
        p1: outer corner
        p2, p3: upper eyelid
        p4: inner corner
        p5, p6: lower eyelid
    
    Reference: "Real-Time Eye Blink Detection using Facial Landmarks"
               Soukupová & Čech, 2016
    
    Args:
        landmarks_px: All face landmarks in pixel coordinates
        eye_indices: 6 indices for eye landmarks
    
    Returns:
        EAR value (typically 0.2-0.4 for open eyes, <0.2 for closed)
    """
    # Get eye landmarks
    eye = landmarks_px[eye_indices].astype(np.float64)
    
    # p1, p2, p3, p4, p5, p6
    p1, p2, p3, p4, p5, p6 = eye
    
    # Vertical distances
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    
    # Horizontal distance
    horizontal = np.linalg.norm(p1 - p4)
    
    # EAR formula
    if horizontal == 0:
        return 0.0
    
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    
    return ear


def compute_gaze_direction(
    left_iris: Optional[Tuple[int, int]],
    right_iris: Optional[Tuple[int, int]],
    left_eye_center: Tuple[int, int],
    right_eye_center: Tuple[int, int]
) -> Tuple[float, float]:
    """
    Compute gaze direction from iris position.
    
    Returns normalized gaze direction (-1 to 1):
        x: -1 = looking left, 0 = center, 1 = looking right
        y: -1 = looking up, 0 = center, 1 = looking down
    
    Args:
        left_iris: Left iris center (x, y) or None
        right_iris: Right iris center (x, y) or None
        left_eye_center: Center of left eye bounding box
        right_eye_center: Center of right eye bounding box
    
    Returns:
        (gaze_x, gaze_y) normalized to [-1, 1]
    """
    if left_iris is None or right_iris is None:
        return (0.0, 0.0)
    
    # Compute relative iris position in each eye
    left_iris = np.array(left_iris, dtype=np.float64)
    right_iris = np.array(right_iris, dtype=np.float64)
    left_center = np.array(left_eye_center, dtype=np.float64)
    right_center = np.array(right_eye_center, dtype=np.float64)
    
    # Normalized offset from eye center
    # Assuming eye width/height is roughly known
    # This is approximate; for better accuracy, use eye dimensions
    
    # Horizontal gaze (left/right)
    gaze_x_left = (left_iris[0] - left_center[0]) / 30.0  # Approximate eye width
    gaze_x_right = (right_iris[0] - right_center[0]) / 30.0
    gaze_x = (gaze_x_left + gaze_x_right) / 2.0
    
    # Vertical gaze (up/down)
    gaze_y_left = (left_iris[1] - left_center[1]) / 15.0  # Approximate eye height
    gaze_y_right = (right_iris[1] - right_center[1]) / 15.0
    gaze_y = (gaze_y_left + gaze_y_right) / 2.0
    
    # Clamp to [-1, 1]
    gaze_x = np.clip(gaze_x, -1.0, 1.0)
    gaze_y = np.clip(gaze_y, -1.0, 1.0)
    
    return (float(gaze_x), float(gaze_y))


# ── Testing ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test the face detector."""
    logger.info("Testing FaceDetector...")
    
    # Create detector
    detector = FaceDetector(refine_landmarks=True)
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Could not open camera")
        detector.close()
        exit()
    
    logger.info("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror for selfie view
        frame = cv2.flip(frame, 1)
        
        # Detect face
        face = detector.detect(frame)
        
        if face:
            # Draw mesh
            frame = detector.draw_mesh(frame, face)
            
            # Draw bounding box
            x, y, w, h = face.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 150), 2)
            
            # Compute and display EAR
            left_ear = compute_ear(face.landmarks_px, face.LEFT_EYE_EAR)
            right_ear = compute_ear(face.landmarks_px, face.RIGHT_EYE_EAR)
            avg_ear = (left_ear + right_ear) / 2
            
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
            
            # Display gaze direction
            if face.left_iris_center:
                left_eye_center = face.landmarks_px[33].astype(int)
                right_eye_center = face.landmarks_px[263].astype(int)
                
                gaze_x, gaze_y = compute_gaze_direction(
                    face.left_iris_center, face.right_iris_center,
                    tuple(left_eye_center), tuple(right_eye_center)
                )
                
                cv2.putText(frame, f"Gaze: ({gaze_x:.2f}, {gaze_y:.2f})", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
        
        cv2.imshow("Face Mesh Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    logger.success("Test complete!")
