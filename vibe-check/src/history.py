"""
Circular buffer for timeline data.
Stores last N seconds of all psychological metrics.
"""
from collections import deque
import time
from typing import Any, Union
from dataclasses import is_dataclass, asdict


class HistoryBuffer:
    def __init__(self, keys: list[str], maxlen: int = 600):
        self.buffers = {k: deque(maxlen=maxlen) for k in keys}
        self.timestamps = deque(maxlen=maxlen)

    def push(self, scores: Union[dict, Any], timestamp: float = None):
        """Push new scores to the buffer. Handles both dicts and dataclasses."""
        self.timestamps.append(timestamp or time.time())
        
        # Convert dataclass to dict if necessary
        if is_dataclass(scores):
            data = asdict(scores)
        elif isinstance(scores, dict):
            data = scores
        else:
            # Fallback for other objects (like classes with __dict__)
            data = getattr(scores, '__dict__', {})
            
        for k in self.buffers:
            self.buffers[k].append(data.get(k, 0.0))

    def get_all(self) -> dict[str, list]:
        return {k: list(v) for k, v in self.buffers.items()}

    def get(self, key: str) -> list:
        return list(self.buffers.get(key, []))
