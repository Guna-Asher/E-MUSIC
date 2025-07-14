# 🎵 E-MUSIC: Emotion-Based Music Recommendation System

**E-MUSIC** is an intelligent music recommendation system that suggests songs based on the user's emotional state. It uses facial emotion recognition to classify mood and then recommends music accordingly.

---

## 🚀 Features

- Detects emotions from facial expressions
- Recommends music tailored to the detected mood
- Interactive and user-friendly interface
- Modular design for easy integration and expansion

---

## 🧠 Tech Stack

- Python
- Jupyter Notebook
- OpenCV (for face detection)
- Deep Learning (CNN for emotion classification)
- Tkinter / Streamlit (optional for GUI)
- Pandas, NumPy, Matplotlib

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Guna-Asher/E-MUSIC.git
   cd E-MUSIC
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖋️ Usage

Run the notebook or script to start the emotion detection and music recommendation:

```bash
jupyter notebook E-MUSIC.ipynb
```

Or if you have a GUI version:

```bash
python app.py
```

---

## 📁 Project Structure

```
E-MUSIC/
├── E-MUSIC.ipynb            # Main notebook
├── emotion_model/           # Trained model files
├── music_dataset/           # Music categorized by emotion
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

---

## 📸 Sample Flow

1. Capture user's face via webcam
2. Detect emotion (e.g., Happy, Sad, Angry)
3. Recommend a playlist or song matching the mood

---

## 🙌 Acknowledgments

- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [OpenCV](https://opencv.org/)
- [Keras](https://keras.io/)
- [Spotify API](https://developer.spotify.com/) *(optional for integration)*

---
