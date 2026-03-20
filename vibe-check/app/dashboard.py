"""
Real-time Streamlit dashboard.
Optimized for high-FPS camera streaming and a professional UI.
"""
import streamlit as st
import numpy as np
import cv2
import time
from src.face_detector import FaceDetector
from src.emotion_engine import EmotionEngine
from models.attention.eye_analyzer import EyeAnalyzer
from src.composite_scorer import compute_all
from src.history import HistoryBuffer
from src.config import settings
from app.camera import CameraThread


def score_color(value: float, inverse: bool = False) -> str:
    """Map 0-1 score to professional color palette."""
    if inverse:
        value = 1.0 - value
    if value < 0.35: return '#ff4b4b'   # Coral Red
    if value < 0.65: return '#faca2b'   # Premium Yellow
    return '#00d289'                    # Mint Green


def build_ui_html(scores) -> str:
    """Builds the entire right-column UI as a single HTML block for maximum FPS."""
    
    # --- Emotio Probs ---
    emotions_html = ""
    # Sort emotions by probability to bring highest to top
    sorted_emotions = sorted(scores.emotion_probs.items(), key=lambda x: x[1], reverse=True)
    for em_name, prob in sorted_emotions:
        width_pct = int(prob * 100)
        emotions_html += f"""
        <div class="emotion-row">
            <div class="emotion-label">{em_name.upper()}</div>
            <div class="emotion-bar-bg">
                <div class="emotion-bar-fill" style="width: {width_pct}%;"></div>
            </div>
            <div class="emotion-value">{width_pct}%</div>
        </div>
        """
        
    # --- Psychological Metrics ---
    metrics_data = [
        ("Valence",    (scores.valence + 1) / 2),
        ("Arousal",     scores.arousal),
        ("Stress",      scores.stress),
        ("Fatigue",     scores.fatigue),
        ("Attention",   scores.attention),
        ("Engagement",  scores.engagement),
    ]
    
    metrics_html = ""
    for label, val in metrics_data:
        val_pct = int(val * 100)
        inverse = label in ["Stress", "Fatigue", "Arousal"] # High stress/fatigue is bad=red
        # For valence, attention, engagement -> low is bad=red
        color = score_color(val, inverse=inverse)
        
        metrics_html += f"""
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-name">{label.upper()}</span>
                <span class="metric-val" style="color: {color};">{val_pct}%</span>
            </div>
            <div class="metric-progress-bg">
                <div class="metric-progress-fill" style="width: {val_pct}%; background-color: {color};"></div>
            </div>
        </div>
        """
        
    html = f"""
    <div class="ui-container">
        <div class="mood-box">
            <div class="mood-label">{scores.mood_label.upper()}</div>
            <div class="mood-sub">DETECTED STATE</div>
        </div>
        
        <div class="section-title">PSYCHOLOGICAL METRICS</div>
        <div class="metrics-grid">
            {metrics_html}
        </div>
        
        <div class="section-title">EMOTION PROBABILITIES</div>
        <div class="emotions-list">
            {emotions_html}
        </div>
    </div>
    """
    return html


def run_dashboard():
    st.set_page_config(
        page_title="Vibe Check",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Clean, professional dark theme
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}

    .ui-container {
        display: flex;
        flex-direction: column;
        gap: 20px;
        padding-top: 10px;
    }

    .mood-box {
        background: #161b22;
        border-radius: 12px;
        border: 1px solid #30363d;
        padding: 30px 20px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }

    .mood-label {
        font-size: 2.2rem;
        font-weight: 600;
        color: #58a6ff;
        letter-spacing: 2px;
    }

    .mood-sub {
        font-size: 0.75rem;
        color: #8b949e;
        letter-spacing: 1.5px;
        margin-top: 8px;
    }

    .section-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #8b949e;
        letter-spacing: 1.2px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
        margin-top: 10px;
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
    }

    .metric-card {
        background: #161b22;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #30363d;
    }

    .metric-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 12px;
        align-items: center;
    }

    .metric-name {
        font-size: 0.75rem;
        color: #c9d1d9;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .metric-val {
        font-size: 0.85rem;
        font-weight: 600;
    }

    .metric-progress-bg {
        width: 100%;
        height: 4px;
        background: #0d1117;
        border-radius: 2px;
        overflow: hidden;
    }

    .metric-progress-fill {
        height: 100%;
        transition: width 0.1s ease;
    }

    .emotions-list {
        display: flex;
        flex-direction: column;
        gap: 12px;
        background: #161b22;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #30363d;
    }

    .emotion-row {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .emotion-label {
        width: 80px;
        font-size: 0.75rem;
        font-weight: 500;
        color: #8b949e;
    }

    .emotion-bar-bg {
        flex-grow: 1;
        height: 6px;
        background: #0d1117;
        border-radius: 3px;
        overflow: hidden;
    }

    .emotion-bar-fill {
        height: 100%;
        background: #58a6ff;
        transition: width 0.1s ease;
    }

    .emotion-value {
        width: 40px;
        text-align: right;
        font-size: 0.75rem;
        color: #c9d1d9;
        font-family: ui-monospace, SFMono-Regular, monospace;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize components
    if "initialized" not in st.session_state:
        st.session_state.camera    = CameraThread()
        st.session_state.detector  = FaceDetector()
        st.session_state.emotion   = EmotionEngine()
        st.session_state.eyes      = EyeAnalyzer()
        st.session_state.history   = HistoryBuffer(
            keys=["valence","arousal","stress","fatigue","attention","engagement"],
            maxlen=settings.history_seconds * settings.history_fps
        )
        st.session_state.camera.start()
        st.session_state.initialized = True

    cam     = st.session_state.camera
    detect  = st.session_state.detector
    emotion = st.session_state.emotion
    eyes    = st.session_state.eyes
    hist    = st.session_state.history

    st.markdown("<h2 style='font-size: 1.5rem; font-weight: 600; color: #c9d1d9; padding-bottom: 20px; border-bottom: 1px solid #30363d; margin-bottom: 25px;'>Vibe Check Engine</h2>", unsafe_allow_html=True)

    # Layout: Camera on left (3), UI on right (2)
    col_cam, col_ui = st.columns([3, 2], gap="large")

    with col_cam:
        cam_placeholder = st.empty()

    with col_ui:
        ui_placeholder = st.empty()

    # Real-time inference loop
    while True:
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        t = time.time()
        face = detect.detect(frame)

        if face is not None:
            # Render camera annotations
            frame = detect.draw_mesh(frame, face, color=(88, 166, 255), thickness=1)
            x, y, w, h = face.bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (88, 166, 255), 2)

            # AI Inference
            probs = emotion.predict(frame, face)
            eye_data = eyes.update(face.landmarks_px, t)
            scores = compute_all(probs, eye_data, list(eyes.ear_history))
            hist.push(scores, t)

            # Update UI Panels (Ultra-fast single DOM update)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(rgb, use_container_width=True)
            
            ui_html = build_ui_html(scores)
            ui_placeholder.markdown(ui_html, unsafe_allow_html=True)

        else:
            # Idle State handling
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(rgb, use_container_width=True)
            ui_placeholder.markdown("""
                <div class="mood-box" style="margin-top: 20px; border-color: rgba(255,75,75,0.3);">
                    <div class="mood-label" style="color: #ff4b4b; font-size: 1.5rem;">NO FACE DETECTED</div>
                    <div class="mood-sub">AWAITING SUBJECT IN FRAME</div>
                </div>
            """, unsafe_allow_html=True)

        time.sleep(0.02)  # Cap at ~50 fps
