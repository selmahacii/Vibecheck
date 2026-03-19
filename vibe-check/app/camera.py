"""
Camera thread: runs OpenCV in a background thread,
puts frames in a queue for the main pipeline to consume.
Threading prevents camera I/O from blocking the UI.
"""
import cv2
import threading
import queue
from loguru import logger
from src.config import settings


class CameraThread:
    def __init__(self):
        self.cap = cv2.VideoCapture(settings.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  settings.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, settings.target_fps)
        self.frame_queue = queue.Queue(maxsize=2)  # small buffer
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera thread started")

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Camera read failed")
                continue
            frame = cv2.flip(frame, 1)  # mirror for selfie mode
            # Drop old frame if queue full (prefer freshness over completeness)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.cap.release()
