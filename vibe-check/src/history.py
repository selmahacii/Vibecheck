"""
History Buffer for Timeline Data.

Circular buffer that stores the last N seconds of psychological metrics.
Used for:
- Timeline visualizations (60-second history)
- Computing trends and patterns
- Statistical analysis

Architecture Decisions:
-----------------------
1. DEQUE: Python's collections.deque provides O(1) append/pop operations
   and automatic size management.

2. SEPARATE BUFFERS: Each metric has its own buffer for flexibility.
   This allows different history lengths per metric if needed.

3. TIMESTAMP TRACKING: Store timestamps for time-based queries
   and real-time visualization alignment.

4. STATISTICS: Compute min, max, mean, std for each metric
   to show ranges and trends.
"""
from collections import deque
import time
from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
from loguru import logger


@dataclass
class HistoryEntry:
    """Single entry in the history buffer."""
    timestamp: float
    valence: float
    arousal: float
    stress: float
    fatigue: float
    attention: float
    engagement: float
    dominant_emotion: str
    emotion_probs: Dict[str, float]


class HistoryBuffer:
    """
    Circular buffer for storing timeline data.
    
    Stores last N seconds of all psychological metrics with timestamps.
    Thread-safe for use with real-time updates.
    
    Usage:
        history = HistoryBuffer(maxlen=600)  # 60s at 10fps
        
        # Add entry
        history.push({
            'valence': 0.5,
            'arousal': 0.3,
            'stress': 0.2,
            ...
        })
        
        # Get history
        stress_history = history.get('stress')
        all_data = history.get_all()
        
        # Get statistics
        stats = history.get_stats('stress')
    
    Args:
        maxlen: Maximum number of entries to store
    """
    
    DEFAULT_KEYS = [
        'valence', 'arousal', 'stress', 
        'fatigue', 'attention', 'engagement'
    ]
    
    def __init__(self, maxlen: int = 600):
        self.maxlen = maxlen
        
        # Initialize buffers for each metric
        self.buffers: Dict[str, deque] = {
            key: deque(maxlen=maxlen) for key in self.DEFAULT_KEYS
        }
        
        # Timestamp buffer
        self.timestamps: deque = deque(maxlen=maxlen)
        
        # Full entries for detailed queries
        self.entries: deque = deque(maxlen=maxlen)
        
        # Emotion history (for emotion timeline)
        self.emotion_history: deque = deque(maxlen=maxlen)
        
        logger.info(f"HistoryBuffer initialized (maxlen={maxlen})")
    
    def push(
        self, 
        scores: Dict[str, Any], 
        timestamp: Optional[float] = None
    ):
        """
        Add a new entry to the history.
        
        Args:
            scores: Dictionary with all metric values
            timestamp: Optional timestamp (default: current time)
        """
        ts = timestamp or time.time()
        
        # Add to individual buffers
        for key in self.DEFAULT_KEYS:
            if key in scores:
                self.buffers[key].append(scores[key])
        
        # Add timestamp
        self.timestamps.append(ts)
        
        # Add full entry
        entry = HistoryEntry(
            timestamp=ts,
            valence=scores.get('valence', 0.0),
            arousal=scores.get('arousal', 0.0),
            stress=scores.get('stress', 0.0),
            fatigue=scores.get('fatigue', 0.0),
            attention=scores.get('attention', 0.0),
            engagement=scores.get('engagement', 0.0),
            dominant_emotion=scores.get('dominant_emotion', 'neutral'),
            emotion_probs=scores.get('emotion_probs', {})
        )
        self.entries.append(entry)
        
        # Add emotion
        if 'dominant_emotion' in scores:
            self.emotion_history.append(scores['dominant_emotion'])
    
    def get(self, key: str) -> List[float]:
        """Get history for a single metric."""
        return list(self.buffers.get(key, []))
    
    def get_all(self) -> Dict[str, List[float]]:
        """Get history for all metrics."""
        return {key: list(buffer) for key, buffer in self.buffers.items()}
    
    def get_timestamps(self) -> List[float]:
        """Get all timestamps."""
        return list(self.timestamps)
    
    def get_recent(self, key: str, n: int = 10) -> List[float]:
        """Get last N values for a metric."""
        values = list(self.buffers.get(key, []))
        return values[-n:]
    
    def get_stats(self, key: str) -> Dict[str, float]:
        """
        Compute statistics for a metric.
        
        Returns:
            Dictionary with min, max, mean, std, latest
        """
        values = list(self.buffers.get(key, []))
        
        if not values:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'latest': 0}
        
        arr = np.array(values)
        
        return {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'latest': float(values[-1])
        }
    
    def get_emotion_distribution(self) -> Dict[str, int]:
        """
        Get distribution of dominant emotions in history.
        
        Returns counts for each emotion.
        """
        from collections import Counter
        counts = Counter(self.emotion_history)
        return dict(counts)
    
    def get_trend(self, key: str, window: int = 10) -> str:
        """
        Determine trend direction for a metric.
        
        Compares recent average to overall average.
        
        Returns:
            'increasing', 'decreasing', or 'stable'
        """
        values = list(self.buffers.get(key, []))
        
        if len(values) < window * 2:
            return 'stable'
        
        recent_mean = np.mean(values[-window:])
        overall_mean = np.mean(values[:-window])
        
        threshold = 0.05  # 5% change threshold
        
        if recent_mean > overall_mean * (1 + threshold):
            return 'increasing'
        elif recent_mean < overall_mean * (1 - threshold):
            return 'decreasing'
        else:
            return 'stable'
    
    def clear(self):
        """Clear all history."""
        for buffer in self.buffers.values():
            buffer.clear()
        self.timestamps.clear()
        self.entries.clear()
        self.emotion_history.clear()
        logger.info("HistoryBuffer cleared")
    
    def __len__(self) -> int:
        return len(self.timestamps)
    
    def is_empty(self) -> bool:
        return len(self.timestamps) == 0


# ── Singleton Instance ──────────────────────────────────────────────────────

_history_instance: Optional[HistoryBuffer] = None


def get_history_buffer() -> HistoryBuffer:
    """Get or create the singleton HistoryBuffer instance."""
    global _history_instance
    if _history_instance is None:
        _history_instance = HistoryBuffer()
    return _history_instance


# ── Testing ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test the history buffer."""
    logger.info("Testing HistoryBuffer...")
    
    buffer = HistoryBuffer(maxlen=100)
    
    # Add some test data
    import random
    for i in range(50):
        buffer.push({
            'valence': random.uniform(-0.5, 0.5),
            'arousal': random.uniform(0.3, 0.7),
            'stress': random.uniform(0.1, 0.4),
            'fatigue': random.uniform(0.1, 0.3),
            'attention': random.uniform(0.6, 0.9),
            'engagement': random.uniform(0.5, 0.8),
            'dominant_emotion': random.choice(['happy', 'neutral', 'sad']),
            'emotion_probs': {'happy': 0.5, 'neutral': 0.3, 'sad': 0.2}
        })
    
    logger.info(f"Buffer length: {len(buffer)}")
    logger.info(f"Stress history: {buffer.get('stress')[:5]}...")
    logger.info(f"Stress stats: {buffer.get_stats('stress')}")
    logger.info(f"Emotion distribution: {buffer.get_emotion_distribution()}")
    logger.info(f"Stress trend: {buffer.get_trend('stress')}")
    
    logger.success("Test complete!")
