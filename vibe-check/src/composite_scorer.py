"""
Composite Scorer - Psychological Metrics Calculator.

Combines emotion predictions and eye metrics to compute psychological
state indicators:
- Valence: Positive vs negative emotion
- Arousal: Energy/activation level
- Stress: Tension and pressure indicator
- Fatigue: Tiredness and exhaustion
- Attention: Focus and concentration
- Engagement: Interest and involvement

Architecture Decisions:
-----------------------
1. VALENCE-AROUSAL MODEL: Based on Russell's Circumplex Model of Affect.
   All emotions can be mapped to 2D space (valence × arousal).
   This enables continuous metrics rather than discrete categories.

2. EMOTION WEIGHTS: Each emotion contributes to valence/arousal based on
   psychological research:
   - Happy: high positive valence, moderate arousal
   - Angry: negative valence, high arousal
   - Sad: negative valence, low arousal
   - Fear: negative valence, very high arousal
   - etc.

3. EYE METRIC INTEGRATION: Eye behavior modifies the base scores:
   - High blink rate → increased stress
   - Low EAR → increased fatigue
   - Low gaze stability → decreased attention
   - Eye closure → increased fatigue

4. MOOD LABELS: Discretize the continuous metrics into human-readable labels:
   - Content: +valence, -arousal
   - Excited: +valence, +arousal
   - Stressed: -valence, +arousal
   - Sad: -valence, -arousal
   etc.

Reference:
- Russell, J.A. (1980). "A circumplex model of affect"
- Posner, J. et al. (2005). "The circumplex model of affect"
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


# ── Valence-Arousal Mapping for Emotions ─────────────────────────────────────

# Each emotion maps to (valence, arousal) in [-1, 1] range
# Based on psychological research (Russell's Circumplex Model)
EMOTION_VA_MAP = {
    'angry':    (-0.5,  0.7),   # Negative, high arousal
    'disgust':  (-0.6,  0.2),   # Negative, low-moderate arousal
    'fear':     (-0.7,  0.8),   # Very negative, very high arousal
    'happy':    ( 0.8,  0.4),   # Very positive, moderate arousal
    'sad':      (-0.6, -0.3),   # Negative, low arousal
    'surprise': ( 0.2,  0.9),   # Slightly positive, very high arousal
    'neutral':  ( 0.0,  0.0),   # Center point
}


@dataclass
class CompositeScores:
    """
    Container for all computed psychological metrics.
    
    Attributes:
        emotion_probs: Raw emotion probabilities from CNN
        dominant_emotion: Most likely emotion
        valence: Positive-negative axis (-1 to +1)
        arousal: Energy level (0 to 1)
        stress: Tension indicator (0 to 1)
        fatigue: Tiredness level (0 to 1)
        attention: Focus quality (0 to 1)
        engagement: Interest level (0 to 1)
        mood_label: Human-readable mood description
        mood_quadrant: Which quadrant of VA space
    """
    emotion_probs: Dict[str, float]
    dominant_emotion: str
    valence: float
    arousal: float
    stress: float
    fatigue: float
    attention: float
    engagement: float
    mood_label: str
    mood_quadrant: str


def compute_valence_arousal(emotion_probs: Dict[str, float]) -> Tuple[float, float]:
    """
    Compute valence and arousal from emotion probabilities.
    
    Uses weighted average of emotion VA values:
        valence = Σ(prob_i × valence_i)
        arousal = Σ(prob_i × arousal_i)
    
    Args:
        emotion_probs: Dictionary mapping emotion names to probabilities
    
    Returns:
        (valence, arousal) tuple in [-1, 1] range
    """
    valence = 0.0
    arousal = 0.0
    
    for emotion, prob in emotion_probs.items():
        if emotion in EMOTION_VA_MAP:
            v, a = EMOTION_VA_MAP[emotion]
            valence += prob * v
            arousal += prob * a
    
    # Clamp to valid range
    valence = float(np.clip(valence, -1.0, 1.0))
    arousal = float(np.clip(arousal, -1.0, 1.0))
    
    return valence, arousal


def compute_stress(
    valence: float,
    arousal: float,
    blink_rate: float,
    gaze_stability: float
) -> float:
    """
    Compute stress level from emotion and eye metrics.
    
    Stress is associated with:
    - Negative valence (feeling bad)
    - High arousal (activated state)
    - High blink rate (anxiety indicator)
    - Low gaze stability (can't focus)
    
    Formula:
        stress = w1×(1-valence)/2 + w2×arousal + w3×blink_factor + w4×gaze_factor
    
    Args:
        valence: Emotion valence (-1 to 1)
        arousal: Emotion arousal (-1 to 1, but typically 0-1)
        blink_rate: Blinks per minute
        gaze_stability: Variance of gaze (lower = more stable)
    
    Returns:
        Stress level (0 to 1)
    """
    # Normalize valence to [0, 1] where higher = more negative
    valence_factor = (1 - valence) / 2
    
    # Normalize arousal to [0, 1]
    arousal_factor = (arousal + 1) / 2
    
    # Blink rate factor
    # Normal: 12-20 bpm, Stressed: >30 bpm
    if blink_rate < 5:
        blink_factor = 0.3  # Very low may indicate dissociation
    elif blink_rate < 20:
        blink_factor = blink_rate / 40  # Normal range
    elif blink_rate < 35:
        blink_factor = 0.5 + (blink_rate - 20) / 30
    else:
        blink_factor = 1.0  # Very high blink rate
    
    # Gaze stability factor (higher variance = less stable = more stress)
    gaze_factor = min(1.0, gaze_stability * 3)
    
    # Weighted combination
    stress = (
        0.25 * valence_factor +
        0.30 * arousal_factor +
        0.25 * blink_factor +
        0.20 * gaze_factor
    )
    
    return float(np.clip(stress, 0.0, 1.0))


def compute_fatigue(
    arousal: float,
    ear_avg: float,
    blink_rate: float,
    closure_percent: float
) -> float:
    """
    Compute fatigue level from emotion and eye metrics.
    
    Fatigue is associated with:
    - Low arousal (low energy)
    - Low EAR (eyes closing)
    - Low blink rate (tired people blink less)
    - High closure percentage (eyes closed often)
    
    Formula:
        fatigue = w1×(1-arousal)/2 + w2×ear_factor + w3×blink_factor + w4×closure
    
    Args:
        arousal: Emotion arousal (-1 to 1)
        ear_avg: Average Eye Aspect Ratio
        blink_rate: Blinks per minute
        closure_percent: Percentage of time eyes were closed
    
    Returns:
        Fatigue level (0 to 1)
    """
    # Low arousal indicates fatigue
    arousal_factor = (1 - arousal) / 2
    
    # Low EAR indicates drowsiness
    # Normal EAR: 0.25-0.35, Drowsy: < 0.20
    if ear_avg > 0.25:
        ear_factor = 0.0
    elif ear_avg > 0.15:
        ear_factor = (0.25 - ear_avg) / 0.10
    else:
        ear_factor = 1.0  # Eyes nearly closed
    
    # Low blink rate can indicate fatigue
    # Normal: 12-20 bpm, Fatigued: < 5 bpm
    if blink_rate < 5:
        blink_factor = 1.0 - (blink_rate / 5)
    else:
        blink_factor = 0.0
    
    # High closure percentage directly indicates fatigue
    closure_factor = closure_percent
    
    # Weighted combination
    fatigue = (
        0.20 * arousal_factor +
        0.35 * ear_factor +
        0.15 * blink_factor +
        0.30 * closure_factor
    )
    
    return float(np.clip(fatigue, 0.0, 1.0))


def compute_attention(
    base_attention: float,
    valence: float,
    arousal: float
) -> float:
    """
    Compute attention level from eye metrics and emotion.
    
    High attention is associated with:
    - High eye-based attention (from EyeAnalyzer)
    - Positive valence (engaged)
    - Moderate arousal (alert but not stressed)
    
    Args:
        base_attention: Attention score from EyeAnalyzer (0 to 1)
        valence: Emotion valence (-1 to 1)
        arousal: Emotion arousal (-1 to 1)
    
    Returns:
        Attention level (0 to 1)
    """
    # Positive valence slightly boosts attention
    valence_factor = (valence + 1) / 2
    
    # Moderate arousal is optimal for attention
    # Too low = drowsy, too high = stressed
    optimal_arousal = 0.3  # Sweet spot
    arousal_factor = 1.0 - abs(arousal - optimal_arousal)
    
    # Weighted combination
    attention = (
        0.60 * base_attention +
        0.15 * valence_factor +
        0.25 * arousal_factor
    )
    
    return float(np.clip(attention, 0.0, 1.0))


def compute_engagement(
    valence: float,
    arousal: float,
    attention: float
) -> float:
    """
    Compute engagement level from emotion and attention.
    
    High engagement is associated with:
    - Positive valence (interested, not bored)
    - Moderate-high arousal (alert and activated)
    - High attention (focused)
    
    This captures whether someone is genuinely interested
    versus just going through the motions.
    
    Args:
        valence: Emotion valence (-1 to 1)
        arousal: Emotion arousal (-1 to 1)
        attention: Attention level (0 to 1)
    
    Returns:
        Engagement level (0 to 1)
    """
    # Positive valence is key for engagement
    valence_factor = (valence + 1) / 2
    
    # Moderate arousal is optimal
    # Too low = bored, too high = anxious
    arousal_factor = max(0, 1.0 - abs(arousal - 0.4))
    
    # High attention contributes to engagement
    attention_factor = attention
    
    # Weighted combination
    engagement = (
        0.35 * valence_factor +
        0.25 * arousal_factor +
        0.40 * attention_factor
    )
    
    return float(np.clip(engagement, 0.0, 1.0))


def determine_mood_label(
    valence: float,
    arousal: float,
    stress: float,
    fatigue: float
) -> Tuple[str, str]:
    """
    Determine human-readable mood label from metrics.
    
    Uses the circumplex model quadrants:
    - Q1 (+V, +A): Excited, Elated, Happy
    - Q2 (-V, +A): Stressed, Anxious, Angry
    - Q3 (-V, -A): Sad, Bored, Fatigued
    - Q4 (+V, -A): Calm, Content, Relaxed
    
    Also considers stress and fatigue levels.
    
    Args:
        valence: Emotion valence (-1 to 1)
        arousal: Emotion arousal (-1 to 1)
        stress: Stress level (0 to 1)
        fatigue: Fatigue level (0 to 1)
    
    Returns:
        (mood_label, quadrant) tuple
    """
    # Normalize arousal to [0, 1] for easier thresholding
    arousal_norm = (arousal + 1) / 2
    
    # Override with high stress/fatigue
    if stress > 0.7:
        if valence < -0.3:
            label = "Stressed"
        else:
            label = "Tense"
        quadrant = "high_stress"
    elif fatigue > 0.7:
        label = "Exhausted"
        quadrant = "high_fatigue"
    elif fatigue > 0.5:
        label = "Tired"
        quadrant = "fatigued"
    else:
        # Use circumplex quadrants
        if valence >= 0:
            if arousal_norm >= 0.5:
                quadrant = "Q1_active_positive"
                if arousal_norm > 0.7:
                    label = "Excited"
                elif arousal_norm > 0.6:
                    label = "Energetic"
                else:
                    label = "Happy"
            else:
                quadrant = "Q4_calm_positive"
                if arousal_norm < 0.3:
                    label = "Peaceful"
                else:
                    label = "Content"
        else:
            if arousal_norm >= 0.5:
                quadrant = "Q2_active_negative"
                if arousal_norm > 0.7:
                    label = "Anxious"
                else:
                    label = "Troubled"
            else:
                quadrant = "Q3_calm_negative"
                if arousal_norm < 0.3:
                    label = "Depressed"
                else:
                    label = "Sad"
    
    return label, quadrant


def compute_all(
    emotion_probs: Dict[str, float],
    eye_metrics,
    ear_history: Optional[List[float]] = None
) -> CompositeScores:
    """
    Compute all psychological metrics from emotion and eye data.
    
    Main entry point for the composite scorer.
    
    Args:
        emotion_probs: Emotion probabilities from EmotionEngine
        eye_metrics: EyeMetrics from EyeAnalyzer
        ear_history: Optional list of recent EAR values
    
    Returns:
        CompositeScores with all computed metrics
    """
    # Get dominant emotion
    dominant_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
    
    # Compute valence and arousal
    valence, arousal = compute_valence_arousal(emotion_probs)
    
    # Compute stress
    stress = compute_stress(
        valence, arousal,
        eye_metrics.blink_rate,
        eye_metrics.gaze_stability
    )
    
    # Compute fatigue
    # Use closure from eye metrics, or estimate from EAR history
    closure_percent = eye_metrics.closure_percent
    if ear_history and len(ear_history) > 0:
        # Can also estimate closure from recent EAR values
        recent_closure = sum(1 for ear in ear_history[-100:] if ear < 0.2) / min(100, len(ear_history))
        closure_percent = max(closure_percent, recent_closure)
    
    fatigue = compute_fatigue(
        arousal,
        eye_metrics.ear_avg,
        eye_metrics.blink_rate,
        closure_percent
    )
    
    # Compute attention
    attention = compute_attention(
        eye_metrics.attention_score,
        valence,
        arousal
    )
    
    # Compute engagement
    engagement = compute_engagement(
        valence,
        arousal,
        attention
    )
    
    # Determine mood label
    mood_label, mood_quadrant = determine_mood_label(
        valence, arousal, stress, fatigue
    )
    
    return CompositeScores(
        emotion_probs=emotion_probs,
        dominant_emotion=dominant_emotion,
        valence=valence,
        arousal=arousal,
        stress=stress,
        fatigue=fatigue,
        attention=attention,
        engagement=engagement,
        mood_label=mood_label,
        mood_quadrant=mood_quadrant
    )


# ── Visualization Helpers ────────────────────────────────────────────────────

def get_valence_arousal_emoji(valence: float, arousal: float) -> str:
    """Get emoji representation for VA coordinates."""
    arousal_norm = (arousal + 1) / 2
    
    if valence >= 0:
        if arousal_norm >= 0.6:
            return "😃"  # Excited/Happy
        elif arousal_norm >= 0.4:
            return "😊"  # Content
        else:
            return "😌"  # Calm
    else:
        if arousal_norm >= 0.6:
            return "😰"  # Anxious/Stressed
        elif arousal_norm >= 0.4:
            return "😔"  # Sad
        else:
            return "😴"  # Fatigued


def get_metric_color(value: float, inverse: bool = False) -> str:
    """
    Get color for metric visualization.
    
    Green (good) → Yellow (moderate) → Red (bad)
    
    Args:
        value: Metric value (0 to 1)
        inverse: If True, low values are good (for stress, fatigue)
    
    Returns:
        Hex color string
    """
    if inverse:
        value = 1 - value
    
    if value >= 0.66:
        return "#44ff88"  # Green
    elif value >= 0.33:
        return "#ffcc44"  # Yellow
    else:
        return "#ff4455"  # Red


# ── Testing ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test the composite scorer."""
    from models.attention.eye_analyzer import EyeMetrics
    
    logger.info("Testing Composite Scorer...")
    
    # Test with sample data
    test_cases = [
        # Happy + alert
        {
            'emotion_probs': {'happy': 0.7, 'neutral': 0.2, 'surprise': 0.1},
            'eye_metrics': EyeMetrics(
                ear_left=0.3, ear_right=0.3, ear_avg=0.3,
                is_blink=False, blink_rate=15, blink_count=50,
                gaze_x=0.0, gaze_y=0.0, gaze_stability=0.1,
                attention_score=0.9, closure_percent=0.0
            )
        },
        # Stressed
        {
            'emotion_probs': {'fear': 0.5, 'angry': 0.3, 'sad': 0.2},
            'eye_metrics': EyeMetrics(
                ear_left=0.25, ear_right=0.25, ear_avg=0.25,
                is_blink=False, blink_rate=35, blink_count=100,
                gaze_x=0.2, gaze_y=-0.1, gaze_stability=0.5,
                attention_score=0.5, closure_percent=0.1
            )
        },
        # Fatigued
        {
            'emotion_probs': {'sad': 0.4, 'neutral': 0.4, 'angry': 0.2},
            'eye_metrics': EyeMetrics(
                ear_left=0.15, ear_right=0.15, ear_avg=0.15,
                is_blink=False, blink_rate=5, blink_count=20,
                gaze_x=0.0, gaze_y=0.2, gaze_stability=0.2,
                attention_score=0.3, closure_percent=0.3
            )
        },
    ]
    
    for i, test in enumerate(test_cases):
        logger.info(f"\n--- Test Case {i+1} ---")
        logger.info(f"Emotions: {test['emotion_probs']}")
        
        scores = compute_all(test['emotion_probs'], test['eye_metrics'])
        
        logger.info(f"Dominant emotion: {scores.dominant_emotion}")
        logger.info(f"Valence: {scores.valence:.2f}")
        logger.info(f"Arousal: {scores.arousal:.2f}")
        logger.info(f"Stress: {scores.stress:.2f}")
        logger.info(f"Fatigue: {scores.fatigue:.2f}")
        logger.info(f"Attention: {scores.attention:.2f}")
        logger.info(f"Engagement: {scores.engagement:.2f}")
        logger.info(f"Mood: {scores.mood_label} ({scores.mood_quadrant})")
        logger.info(f"Emoji: {get_valence_arousal_emoji(scores.valence, scores.arousal)}")
    
    logger.success("\nAll tests passed!")
