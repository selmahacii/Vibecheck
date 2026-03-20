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
    sorted_emotions = sorted(scores.emotion_probs.items(), key=lambda x: x[1], reverse=True)
    for em_name, prob in sorted_emotions:
        width_pct = int(prob * 100)
        # Dynamic glow effect for the dominant emotion
        glow = "box-shadow: 0 0 10px rgba(88, 166, 255, 0.5);" if width_pct > 50 else ""
        emotions_html += f"""
        <div class="emotion-row">
            <div class="emotion-label">{em_name.upper()}</div>
            <div class="emotion-bar-bg">
                <div class="emotion-bar-fill" style="width: {width_pct}%; {glow}"></div>
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
        inverse = label in ["Stress", "Fatigue", "Arousal"]
        color = score_color(val, inverse=inverse)
        
        metrics_html += f"""
        <div class="metric-card glass-panel">
            <div class="metric-header">
                <span class="metric-name">{label.upper()}</span>
                <span class="metric-val" style="color: {color}; text-shadow: 0 0 8px {color}80;">{val_pct}%</span>
            </div>
            <div class="metric-progress-bg">
                <div class="metric-progress-fill" style="width: {val_pct}%; background: linear-gradient(90deg, transparent, {color}); box-shadow: 0 0 10px {color};"></div>
            </div>
        </div>
        """
        
    html = f"""
    <div class="ui-container">
        <div class="mood-box glass-panel glowing-border">
            <div class="mood-label">{scores.mood_label.upper()}</div>
            <div class="mood-sub">DETECTED STATE</div>
        </div>
        
        <div class="section-title">PSYCHOLOGICAL METRICS</div>
        <div class="metrics-grid">
            {metrics_html}
        </div>
        
        <div class="section-title">EMOTION PROBABILITIES</div>
        <div class="emotions-list glass-panel">
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

    # Premium glassmorphism dark theme
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Background Gradient */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0a0e17 0%, #050508 100%);
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}

    .ui-container {
        display: flex;
        flex-direction: column;
        gap: 24px;
        padding-top: 10px;
    }

    /* Glassmorphism Classes */
    .glass-panel {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    .mood-box {
        border-radius: 16px;
        padding: 35px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    /* Subtle glowing border effect for the mood box */
    .glowing-border::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(88,166,255,0.6), transparent);
    }

    .mood-label {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(to right, #58a6ff, #a371f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 3px;
        filter: drop-shadow(0 0 12px rgba(88, 166, 255, 0.3));
    }

    .mood-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        letter-spacing: 2px;
        margin-top: 10px;
        font-weight: 500;
        text-transform: uppercase;
    }

    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 1.5px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding-bottom: 10px;
        margin-top: 5px;
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
    }

    .metric-card {
        border-radius: 12px;
        padding: 18px;
        transition: transform 0.2s ease, background 0.2s ease;
    }
    
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.04);
        transform: translateY(-2px);
    }

    .metric-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 14px;
        align-items: center;
    }

    .metric-name {
        font-size: 0.8rem;
        color: #e2e8f0;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .metric-val {
        font-size: 0.95rem;
        font-weight: 700;
    }

    .metric-progress-bg {
        width: 100%;
        height: 6px;
        background: rgba(0,0,0,0.4);
        border-radius: 3px;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
    }

    .metric-progress-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .emotions-list {
        border-radius: 12px;
        padding: 24px;
        display: flex;
        flex-direction: column;
        gap: 14px;
    }

    .emotion-row {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .emotion-label {
        width: 85px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #cbd5e1;
        letter-spacing: 0.5px;
    }

    .emotion-bar-bg {
        flex-grow: 1;
        height: 8px;
        background: rgba(0,0,0,0.4);
        border-radius: 4px;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
    }

    .emotion-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #58a6ff);
        border-radius: 4px;
        transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .emotion-value {
        width: 45px;
        text-align: right;
        font-size: 0.85rem;
        font-weight: 600;
        color: #e2e8f0;
        font-family: ui-monospace, SFMono-Regular, monospace;
    }
    
    /* Global st adjustments */
    .stImage img {
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
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

    logo_svg = """
    <svg width="36" height="36" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0px 0px 8px rgba(88, 166, 255, 0.4));">
      <defs>
        <linearGradient id="primaryGrad" x1="0" y1="0" x2="48" y2="48">
          <stop stop-color="#a371f7"/>
          <stop offset="1" stop-color="#58a6ff"/>
        </linearGradient>
        <linearGradient id="secondaryGrad" x1="48" y1="0" x2="0" y2="48">
          <stop stop-color="#58a6ff"/>
          <stop offset="1" stop-color="rgba(88, 166, 255, 0.2)"/>
        </linearGradient>
      </defs>
      <!-- V shape -->
      <path d="M12 14 L24 38 L36 14" stroke="url(#primaryGrad)" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
      <!-- Checkmark shape interlocking -->
      <path d="M18 24 L26 32 L42 8" stroke="url(#secondaryGrad)" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
      <!-- Pulsing dot -->
      <circle cx="24" cy="38" r="4" fill="#ffffff"/>
    </svg>
    """

    st.markdown(f"""
        <div style='display: flex; align-items: center; gap: 14px; padding-bottom: 16px; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 24px;'>
            {{logo_svg}}
            <h2 style='font-size: 1.8rem; font-weight: 700; color: #e2e8f0; margin: 0; letter-spacing: 1.5px;'>
                Vibe<span style='color: #58a6ff; font-weight: 300;'>Check</span>
            </h2>
        </div>
    """.replace("{logo_svg}", logo_svg), unsafe_allow_html=True)

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
