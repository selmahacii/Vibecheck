"""
Eye Analyzer for Attention and Blink Detection.

Analyzes eye behavior using MediaPipe FaceMesh landmarks to compute:
- Blink rate (blinks per minute)
- Blink duration
- Eye closure percentage
- Gaze stability
- Attention score

These metrics are combined with emotion predictions to derive
psychological state indicators like stress and fatigue.

Architecture Decisions:
-----------------------
1. EYE ASPECT RATIO (EAR): Standard method for blink detection.
   EAR drops significantly when eyes close.
   Threshold: 0.2 (tunable per individual)

2. TEMPORAL SMOOTHING: Use exponential moving average for EAR
   to reduce noise from landmark detection jitter.

3. BLINK DETECTION: Track EAR over time, detect when:
   - EAR drops below threshold (blink start)
   - EAR rises above threshold (blink end)
   - Duration < 0.5 seconds (avoid false positives from looking down)

4. HISTORY BUFFER: Store last N seconds of EAR values for:
   - Computing blink rate
   - Detecting prolonged closure (fatigue indicator)

5. GAZE STABILITY: Track iris position variance over time.
   High variance indicates distraction; low variance indicates focus.

Reference:
- Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks", 2016
"""
import numpy as np
from collections import deque
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class EyeMetrics:
    """
    Container for computed eye metrics.
    
    Attributes:
        ear_left: Eye Aspect Ratio for left eye
        ear_right: Eye Aspect Ratio for right eye
        ear_avg: Average EAR across both eyes
        is_blink: Whether a blink is currently happening
        blink_rate: Blinks per minute (computed over last 60s)
        blink_count: Total blinks in current session
        gaze_x: Horizontal gaze direction (-1 left, 0 center, 1 right)
        gaze_y: Vertical gaze direction (-1 up, 0 center, 1 down)
        gaze_stability: Variance of gaze over time (lower = more stable)
        attention_score: Computed attention level (0-1)
        closure_percent: Percentage of time eyes were closed
    """
    ear_left: float
    ear_right: float
    ear_avg: float
    is_blink: bool
    blink_rate: float
    blink_count: int
    gaze_x: float
    gaze_y: float
    gaze_stability: float
    attention_score: float
    closure_percent: float


class EyeAnalyzer:
    """
    Analyzes eye behavior for attention and fatigue detection.
    
    Uses MediaPipe FaceMesh landmarks to compute:
    - Eye Aspect Ratio (EAR) for blink detection
    - Blink rate and duration
    - Gaze direction and stability
    - Overall attention score
    
    Usage:
        analyzer = EyeAnalyzer()
        
        # In your main loop:
        for frame in video:
            face = detector.detect(frame)
            if face:
                metrics = analyzer.update(face.landmarks_px, timestamp)
                print(f"Attention: {metrics.attention_score:.2f}")
                print(f"Blink rate: {metrics.blink_rate:.1f} bpm")
    
    Args:
        ear_threshold: EAR below which eyes are considered closed
        blink_window_sec: Time window for computing blink rate
        history_duration: Duration of history buffer in seconds
        fps: Expected frames per second
    """
    
    # Standard landmark indices for EAR calculation
    # From MediaPipe FaceMesh documentation
    LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
    
    # Iris landmark indices (if refine_landmarks was True)
    LEFT_IRIS_IDX = list(range(468, 478))
    RIGHT_IRIS_IDX = list(range(478, 488))
    
    # Eye corner indices for gaze normalization
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_INNER = 362
    
    def __init__(
        self,
        ear_threshold: float = 0.2,
        blink_window_sec: float = 60.0,
        history_duration: float = 60.0,
        fps: float = 30.0
    ):
        self.ear_threshold = ear_threshold
        self.blink_window_sec = blink_window_sec
        self.history_duration = history_duration
        self.fps = fps
        
        # History buffers
        history_size = int(history_duration * fps)
        self.ear_history: deque = deque(maxlen=history_size)
        self.gaze_history: deque = deque(maxlen=history_size)
        self.blink_timestamps: deque = deque(maxlen=100)  # Store last 100 blinks
        
        # State tracking
        self.currently_blinking = False
        self.blink_start_time: Optional[float] = None
        self.total_blinks = 0
        self.last_timestamp = 0.0
        
        # EAR smoothing (exponential moving average)
        self.ear_ema = 0.3
        self.smoothed_ear_left = 0.0
        self.smoothed_ear_right = 0.0
        
        # Gaze smoothing
        self.gaze_ema = 0.3
        self.smoothed_gaze_x = 0.0
        self.smoothed_gaze_y = 0.0
        
        logger.info(f"EyeAnalyzer initialized (EAR threshold={ear_threshold})")
    
    def compute_ear(self, landmarks_px: np.ndarray, eye_indices: List[int]) -> float:
        """
        Compute Eye Aspect Ratio for a single eye.
        
        EAR measures eye openness:
        - Open eye: 0.25-0.40
        - Half-closed: 0.15-0.25
        - Closed: < 0.15
        
        Formula: EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        eye = landmarks_px[eye_indices].astype(np.float64)
        p1, p2, p3, p4, p5, p6 = eye
        
        # Vertical distances
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        
        # Horizontal distance
        h = np.linalg.norm(p1 - p4)
        
        if h < 1e-6:
            return 0.0
        
        return (v1 + v2) / (2.0 * h)
    
    def compute_gaze(
        self, 
        landmarks_px: np.ndarray,
        has_iris: bool = True
    ) -> Tuple[float, float]:
        """
        Compute gaze direction from iris position.
        
        Returns normalized values (-1 to 1):
        - gaze_x: -1 (left), 0 (center), 1 (right)
        - gaze_y: -1 (up), 0 (center), 1 (down)
        
        If iris landmarks are not available, falls back to
        estimating from eye corner positions.
        """
        if has_iris and len(landmarks_px) >= 488:
            # Use iris landmarks (more accurate)
            left_iris = landmarks_px[self.LEFT_IRIS_IDX].mean(axis=0)
            right_iris = landmarks_px[self.RIGHT_IRIS_IDX].mean(axis=0)
            
            # Eye centers (using inner and outer corners)
            left_eye_center = landmarks_px[[self.LEFT_EYE_OUTER, self.LEFT_EYE_INNER]].mean(axis=0)
            right_eye_center = landmarks_px[[self.RIGHT_EYE_OUTER, self.RIGHT_EYE_INNER]].mean(axis=0)
            
            # Eye dimensions (approximate)
            left_eye_width = np.linalg.norm(
                landmarks_px[self.LEFT_EYE_OUTER] - landmarks_px[self.LEFT_EYE_INNER]
            )
            right_eye_width = np.linalg.norm(
                landmarks_px[self.RIGHT_EYE_OUTER] - landmarks_px[self.RIGHT_EYE_INNER]
            )
            avg_eye_width = (left_eye_width + right_eye_width) / 2
            
            if avg_eye_width < 1e-6:
                return (0.0, 0.0)
            
            # Normalized offset from center
            gaze_x_left = (left_iris[0] - left_eye_center[0]) / (avg_eye_width * 0.5)
            gaze_x_right = (right_iris[0] - right_eye_center[0]) / (avg_eye_width * 0.5)
            gaze_x = (gaze_x_left + gaze_x_right) / 2
            
            # Approximate eye height as 1/3 of width
            avg_eye_height = avg_eye_width / 3
            gaze_y_left = (left_iris[1] - left_eye_center[1]) / avg_eye_height
            gaze_y_right = (right_iris[1] - right_eye_center[1]) / avg_eye_height
            gaze_y = (gaze_y_left + gaze_y_right) / 2
            
        else:
            # Fallback: estimate from eye shape
            # This is less accurate but works without iris landmarks
            left_eye = landmarks_px[self.LEFT_EYE_EAR_IDX]
            right_eye = landmarks_px[self.RIGHT_EYE_EAR_IDX]
            
            left_center = left_eye.mean(axis=0)
            right_center = right_eye.mean(axis=0)
            
            # Can't determine horizontal gaze without iris
            gaze_x = 0.0
            
            # Vertical gaze: compare upper and lower eyelid positions
            # If upper eyelid is lower than normal, looking down
            gaze_y = 0.0
        
        # Clamp to [-1, 1]
        gaze_x = float(np.clip(gaze_x, -1.0, 1.0))
        gaze_y = float(np.clip(gaze_y, -1.0, 1.0))
        
        return (gaze_x, gaze_y)
    
    def detect_blink(
        self,
        ear_avg: float,
        timestamp: float
    ) -> Tuple[bool, bool]:
        """
        Detect blink based on EAR value.
        
        Uses threshold-based detection with hysteresis:
        - Blink starts when EAR drops below threshold
        - Blink ends when EAR rises above threshold
        
        Args:
            ear_avg: Current average EAR
            timestamp: Current timestamp in seconds
        
        Returns:
            (is_blink, blink_ended): Current blink state and whether a blink just ended
        """
        blink_ended = False
        
        if ear_avg < self.ear_threshold:
            # Eyes are closed
            if not self.currently_blinking:
                # Blink started
                self.currently_blinking = True
                self.blink_start_time = timestamp
        else:
            # Eyes are open
            if self.currently_blinking:
                # Blink ended
                self.currently_blinking = False
                blink_ended = True
                
                # Check blink duration (avoid false positives)
                if self.blink_start_time is not None:
                    duration = timestamp - self.blink_start_time
                    
                    # Normal blink: 0.1-0.4 seconds
                    if 0.05 < duration < 0.5:
                        self.total_blinks += 1
                        self.blink_timestamps.append(timestamp)
                        logger.debug(f"Blink detected (duration={duration:.2f}s)")
        
        return (self.currently_blinking, blink_ended)
    
    def compute_blink_rate(self, current_time: float) -> float:
        """
        Compute blink rate (blinks per minute).
        
        Counts blinks within the configured time window.
        """
        # Filter blinks within window
        cutoff_time = current_time - self.blink_window_sec
        recent_blinks = [
            t for t in self.blink_timestamps 
            if t > cutoff_time
        ]
        
        # Blinks per minute
        blink_rate = len(recent_blinks) * (60.0 / self.blink_window_sec)
        
        return blink_rate
    
    def compute_gaze_stability(self) -> float:
        """
        Compute gaze stability (variance of gaze direction).
        
        Lower values = more stable gaze = higher attention.
        Higher values = wandering gaze = lower attention.
        """
        if len(self.gaze_history) < 10:
            return 0.0
        
        gazes = np.array(list(self.gaze_history))
        variance = np.var(gazes, axis=0).sum()
        
        return float(variance)
    
    def compute_attention_score(
        self,
        ear_avg: float,
        blink_rate: float,
        gaze_stability: float
    ) -> float:
        """
        Compute overall attention score.
        
        Factors:
        1. Eye openness (higher EAR = more alert)
        2. Normal blink rate (12-20 bpm is healthy)
        3. Gaze stability (lower variance = more focused)
        
        Returns score in [0, 1] where 1 = highly attentive.
        """
        # EAR component (0.3 = open, 0.1 = closed)
        ear_score = min(1.0, max(0.0, (ear_avg - 0.1) / 0.25))
        
        # Blink rate component
        # Optimal: 12-20 blinks per minute
        # Too few = fatigue, too many = stress/distraction
        if blink_rate < 5:
            blink_score = 0.5  # May indicate fatigue
        elif blink_rate < 12:
            blink_score = 0.8
        elif blink_rate < 20:
            blink_score = 1.0  # Normal
        elif blink_rate < 30:
            blink_score = 0.7
        else:
            blink_score = 0.4  # Too many blinks = stress
        
        # Gaze stability component
        # Lower variance = more focused
        gaze_score = max(0.0, 1.0 - gaze_stability * 2)
        
        # Weighted combination
        attention = (
            ear_score * 0.4 +
            blink_score * 0.3 +
            gaze_score * 0.3
        )
        
        return float(np.clip(attention, 0.0, 1.0))
    
    def compute_closure_percent(self) -> float:
        """
        Compute percentage of time eyes were closed.
        
        High closure percentage indicates fatigue.
        """
        if len(self.ear_history) == 0:
            return 0.0
        
        closed_count = sum(1 for ear in self.ear_history if ear < self.ear_threshold)
        return closed_count / len(self.ear_history)
    
    def update(
        self,
        landmarks_px: np.ndarray,
        timestamp: float,
        fps: Optional[float] = None
    ) -> EyeMetrics:
        """
        Update eye analysis with new frame.
        
        Main entry point for the eye analyzer.
        Call this for each frame with face landmarks.
        
        Args:
            landmarks_px: Face landmarks in pixel coordinates (468+ points)
            timestamp: Current timestamp in seconds
            fps: Current frame rate (optional, uses default if not provided)
        
        Returns:
            EyeMetrics with all computed values
        """
        # Update FPS if provided
        if fps is not None:
            self.fps = fps
        
        # Compute EAR for both eyes
        ear_left = self.compute_ear(landmarks_px, self.LEFT_EYE_EAR_IDX)
        ear_right = self.compute_ear(landmarks_px, self.RIGHT_EYE_EAR_IDX)
        ear_avg = (ear_left + ear_right) / 2
        
        # Smooth EAR with exponential moving average
        if self.smoothed_ear_left == 0:
            self.smoothed_ear_left = ear_left
            self.smoothed_ear_right = ear_right
        else:
            self.smoothed_ear_left = self.ear_ema * ear_left + (1 - self.ear_ema) * self.smoothed_ear_left
            self.smoothed_ear_right = self.ear_ema * ear_right + (1 - self.ear_ema) * self.smoothed_ear_right
        
        smoothed_ear_avg = (self.smoothed_ear_left + self.smoothed_ear_right) / 2
        
        # Store in history
        self.ear_history.append(smoothed_ear_avg)
        
        # Detect blinks
        is_blink, _ = self.detect_blink(smoothed_ear_avg, timestamp)
        
        # Compute blink rate
        blink_rate = self.compute_blink_rate(timestamp)
        
        # Compute gaze
        has_iris = len(landmarks_px) >= 488
        gaze_x, gaze_y = self.compute_gaze(landmarks_px, has_iris)
        
        # Smooth gaze
        self.smoothed_gaze_x = self.gaze_ema * gaze_x + (1 - self.gaze_ema) * self.smoothed_gaze_x
        self.smoothed_gaze_y = self.gaze_ema * gaze_y + (1 - self.gaze_ema) * self.smoothed_gaze_y
        
        # Store gaze in history
        self.gaze_history.append([self.smoothed_gaze_x, self.smoothed_gaze_y])
        
        # Compute metrics
        gaze_stability = self.compute_gaze_stability()
        closure_percent = self.compute_closure_percent()
        attention_score = self.compute_attention_score(
            smoothed_ear_avg, blink_rate, gaze_stability
        )
        
        # Update timestamp
        self.last_timestamp = timestamp
        
        return EyeMetrics(
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=smoothed_ear_avg,
            is_blink=is_blink,
            blink_rate=blink_rate,
            blink_count=self.total_blinks,
            gaze_x=self.smoothed_gaze_x,
            gaze_y=self.smoothed_gaze_y,
            gaze_stability=gaze_stability,
            attention_score=attention_score,
            closure_percent=closure_percent
        )
    
    def reset(self):
        """Reset all history and state."""
        self.ear_history.clear()
        self.gaze_history.clear()
        self.blink_timestamps.clear()
        self.currently_blinking = False
        self.blink_start_time = None
        self.total_blinks = 0
        self.smoothed_ear_left = 0.0
        self.smoothed_ear_right = 0.0
        self.smoothed_gaze_x = 0.0
        self.smoothed_gaze_y = 0.0
        logger.info("EyeAnalyzer reset")


# ── Testing ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test the eye analyzer."""
    import cv2
    import time
    
    # Import face detector
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from face_detector import FaceDetector
    
    logger.info("Testing EyeAnalyzer...")
    
    # Initialize
    detector = FaceDetector(refine_landmarks=True)
    analyzer = EyeAnalyzer()
    
    cap = cv2.VideoCapture(0)
    
    logger.info("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        t = time.time()
        
        face = detector.detect(frame)
        
        if face:
            metrics = analyzer.update(face.landmarks_px, t)
            
            # Draw mesh
            frame = detector.draw_mesh(frame, face)
            
            # Display metrics
            y_offset = 30
            metrics_text = [
                f"EAR: {metrics.ear_avg:.2f}",
                f"Blink: {'Yes' if metrics.is_blink else 'No'}",
                f"Blink Rate: {metrics.blink_rate:.1f} bpm",
                f"Gaze: ({metrics.gaze_x:.2f}, {metrics.gaze_y:.2f})",
                f"Attention: {metrics.attention_score:.2f}",
            ]
            
            for i, text in enumerate(metrics_text):
                cv2.putText(frame, text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2)
        
        cv2.imshow("Eye Analyzer Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    logger.success("Test complete!")
