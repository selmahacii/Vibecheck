"""
Real-time Streamlit dashboard.
Layout:
┌─────────────────────┬───────────────────────┐
│  Live Camera Feed   │   Dominant Emotion     │
│  with face mesh     │   + intensity bar      │
│                     │                        │
├─────────────────────┼────────────┬───────────┤
│  Emotion Radar      │  Timeline  │  Scores   │
│  Chart (7 emotions) │  (60s)     │  Panel    │
└─────────────────────┴────────────┴───────────┘
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
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


def build_radar_chart(probs: dict) -> go.Figure:
    """
    Radar chart showing all 7 emotion probabilities.
    Updates every frame — shows the current emotional state at a glance.
    """
    emotions = list(probs.keys())
    values   = list(probs.values())
    values.append(values[0])  # close the polygon
    emotions.append(emotions[0])

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=emotions,
        fill='toself',
        fillcolor='rgba(0, 255, 150, 0.15)',
        line=dict(color='rgba(0, 255, 150, 0.8)', width=2),
        marker=dict(size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, range=[0, 1],
                color='rgba(255,255,255,0.3)',
                gridcolor='rgba(255,255,255,0.1)',
            ),
            angularaxis=dict(color='rgba(255,255,255,0.6)'),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=280,
    )
    return fig


def build_timeline_chart(history: dict, metric: str,
                          color: str = '#00ff96') -> go.Figure:
    """Line chart showing last 60s of a metric."""
    values = history.get(metric, [])
    times  = list(range(len(values)))

    fig = go.Figure(go.Scatter(
        x=times, y=values,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=color.replace(')', ',0.08)').replace('rgb', 'rgba'),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(range=[0, 1], color='rgba(255,255,255,0.4)',
                   gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(l=30, r=10, t=10, b=10),
        height=120,
    )
    return fig


def score_color(value: float) -> str:
    """Map 0-1 score to hex color: green → yellow → red."""
    if value < 0.33: return '#44ff88'
    if value < 0.66: return '#ffcc44'
    return '#ff4455'


def run_dashboard():
    st.set_page_config(
        page_title="Vibe Check",
        layout="wide",
        page_icon="🧠",
        initial_sidebar_state="collapsed",
    )

    # Dark theme CSS
    st.markdown("""
    <style>
    .stApp { background: #07070f; color: #e0e0e0; }
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .mood-label {
        font-size: 2.5rem;
        text-align: center;
        font-weight: 300;
        letter-spacing: 0.1em;
        padding: 20px 0;
    }
    .score-label {
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.4);
        margin-bottom: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("# 🧠 Vibe Check")

    # Initialize components (cached in session state)
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

    # ── Layout ──────────────────────────────────────────────
    col_cam, col_right = st.columns([3, 2])

    with col_cam:
        cam_placeholder   = st.empty()
        radar_placeholder = st.empty()

    with col_right:
        mood_placeholder    = st.empty()
        scores_placeholder  = st.empty()
        timeline_placeholder = st.empty()

    # ── Real-time loop ───────────────────────────────────────
    while True:
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        t = time.time()
        face = detect.detect(frame)

        if face is not None:
            # Draw face mesh on frame
            frame = detect.draw_mesh(frame, face,
                                     color=(0, 255, 150), thickness=1)
            # Draw bounding box
            x, y, w, h = face.bbox
            cv2.rectangle(frame, (x,y), (x+w,y+h),
                          (0,255,150), 1)

            probs      = emotion.predict(frame, face)
            eye_data   = eyes.update(face.landmarks_px, t)
            scores     = compute_all(probs, eye_data,
                                     list(eyes.ear_history))
            hist.push(scores, t)

            # ── Camera feed ──────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(rgb, use_column_width=True,
                                  caption="Live feed · MediaPipe FaceMesh")

            # ── Mood label ───────────────────────────────────
            mood_placeholder.markdown(
                f'<div class="mood-label">{scores["mood_label"]}</div>',
                unsafe_allow_html=True
            )

            # ── Emotion radar ────────────────────────────────
            radar_placeholder.plotly_chart(
                build_radar_chart(scores["emotion_probs"]),
                use_container_width=True,
                config={"displayModeBar": False}
            )

            # ── Score cards ──────────────────────────────────
            with scores_placeholder.container():
                metrics = [
                    ("Valence",    (scores["valence"]+1)/2, "😊 ↔ 😢"),
                    ("Arousal",     scores["arousal"],      "⚡ énergie"),
                    ("Stress",      scores["stress"],       "😰 tension"),
                    ("Fatigue",     scores["fatigue"],      "😴 épuisement"),
                    ("Attention",   scores["attention"],    "👁 focus"),
                    ("Engagement",  scores["engagement"],   "🎯 engagement"),
                ]
                cols = st.columns(3)
                for i, (label, val, icon) in enumerate(metrics):
                    with cols[i % 3]:
                        color = score_color(
                            val if label not in ["Valence","Attention","Engagement"]
                            else 1-val
                        )
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<div class="score-label">{icon} {label}</div>'
                            f'<div style="font-size:1.6rem;color:{color};'
                            f'font-weight:300">{val*100:.0f}%</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            # ── Timeline (stress last 60s) ───────────────────
            with timeline_placeholder.container():
                st.markdown("**Stress · 60s**")
                st.plotly_chart(
                    build_timeline_chart(
                        hist.get_all(), "stress", '#ff6655'
                    ),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

        else:
            # No face detected
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(rgb, use_column_width=True)
            mood_placeholder.markdown(
                '<div class="mood-label" style="color:rgba(255,255,255,0.2)">'
                '🔍 Aucun visage détecté</div>',
                unsafe_allow_html=True
            )

        time.sleep(0.033)  # ~30fps
