"""
Entry point. Usage:
  # 1. Download dataset
  python data/download_fer2013.py

  # 2. Train the model (GPU recommended, ~2h on CPU)
  python -m models.emotion.train

  # 3. Launch dashboard
  streamlit run app/main.py
"""
from app.dashboard import run_dashboard

if __name__ == "__main__":
    run_dashboard()
