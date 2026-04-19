# 🤚 Sign Language Translator (Real-Time)

> **Accessibility AI** — Detects hand signs via webcam and translates them into text and speech in real time.  
> Helps the deaf/mute community communicate — and looks *extremely* impressive in any AI portfolio.

---

## 📸 Demo Flow

```
Webcam → MediaPipe Hand Landmarks → Neural Network → Letter/Word → Sentence → 🔊 Speech
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **MediaPipe** | Real-time 21-point hand landmark detection |
| **OpenCV** | Webcam capture, drawing, UI overlay |
| **TensorFlow / Keras** | Dense neural network classifier |
| **pyttsx3** | Offline text-to-speech (no internet needed) |
| **NumPy / scikit-learn** | Data handling, label encoding, train/test split |
| **Matplotlib** | Training accuracy/loss plots |

---

## 📁 Project Structure

```
sign-language-translator/
│
├── data_collection.py   ← Step 1: Collect hand-sign images from webcam
├── train_model.py       ← Step 2: Extract landmarks + train classifier
├── app.py               ← Step 3: Real-time translator app
│
├── requirements.txt     ← All Python dependencies
│
├── data/                ← Auto-created by data_collection.py
│   ├── A/  (100 images)
│   ├── B/  (100 images)
│   └── ...
│
└── model/               ← Auto-created by train_model.py
    ├── sign_language_model.h5
    ├── label_map.npy
    └── training_history.png
```

---

## 🚀 Step-by-Step Setup

### Step 0 — Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install all packages
pip install -r requirements.txt
```

---

### Step 1 — Collect Hand-Sign Data 📷

```bash
python data_collection.py
```

**What happens:**
1. Your webcam opens showing a live feed.
2. For each letter/word class (A–Z + SPACE + DEL + NOTHING), the script prompts you.
3. Press **`s`** to start saving 100 images for that class.
4. Show your hand sign clearly in front of the camera.
5. Press **`q`** to skip a class or stop early.

> 💡 **Tips for good data:**
> - Use good lighting (natural light or a lamp facing you).
> - Vary your hand position slightly (slight tilts, distances).
> - Capture ~100 images per class — the script does this automatically.
> - Keep the hand sign stable and centered in the frame.

**Output:** `data/A/`, `data/B/`, … folders with `.jpg` images.

---

### Step 2 — Train the Model 🧠

```bash
python train_model.py
```

**What happens:**
1. Reads all images from `data/`.
2. Uses MediaPipe to extract **21 hand landmarks × (x, y, z) = 63 features** per image.
3. Encodes class labels.
4. Trains a 4-layer Dense Neural Network with Dropout + BatchNorm.
5. Uses EarlyStopping and ReduceLROnPlateau for optimal training.
6. Saves the trained model and label map to `model/`.
7. Saves a training history plot.

**Expected output:**
```
[INFO] Samples: 2800, Features: 63
[INFO] Classes (28): ['A', 'B', ..., 'Z', 'DEL', 'NOTHING', 'SPACE']
...
Epoch 30/30 — val_accuracy: 0.9714
[INFO] Validation accuracy: 97.14%
[INFO] Model saved → model/sign_language_model.h5
```

> 💡 Aim for **>90% validation accuracy**. If lower, collect more varied images.

---

### Step 3 — Run the Real-Time Translator 🎯

```bash
python app.py
```

**What happens:**
1. Webcam opens with a live feed.
2. MediaPipe detects your hand and draws the 21-point skeleton.
3. Every frame, the model predicts which sign you're showing.
4. Prediction is **smoothed over 10 frames** (majority vote) to avoid jitter.
5. A letter is added to the sentence after you **hold the sign for ~20 frames**.
6. The current sentence is shown at the bottom of the screen.

**Keyboard Controls:**

| Key | Action |
|---|---|
| `SPACE` | Add a space between words |
| `ENTER` | 🔊 Speak the current sentence aloud |
| `c` | Clear the sentence buffer |
| `q` | Quit the app |

---

## 🧠 How the AI Works (Explained Simply)

```
Raw Image
    ↓
MediaPipe Hand Landmarks
    → 21 points × (x, y, z) = 63 numbers
    ↓
Dense Neural Network
    → Input(63) → Dense(256) → Dense(128) → Dense(64) → Softmax(28)
    ↓
Predicted Class (e.g. "A")
    ↓
Smoothing (majority vote over 10 frames)
    ↓
Hold detection (20 stable frames → add to sentence)
    ↓
Text sentence → pyttsx3 → 🔊 Speech
```

**Why landmarks instead of raw images?**  
- Much faster (63 numbers vs. 307,200 pixels).  
- Position-invariant — works regardless of where your hand is on screen.  
- Lighter model — runs in real time even on a CPU.

---

## 📊 Model Architecture

```
Input: (63,)
  ↓
Dense(256) + BatchNorm + Dropout(0.4)
  ↓
Dense(128) + BatchNorm + Dropout(0.3)
  ↓
Dense(64) + Dropout(0.2)
  ↓
Dense(num_classes) + Softmax
```

---

## 🔧 Customizing Classes

To recognize **custom words/gestures** instead of letters, edit `data_collection.py`:

```python
# Replace with your own classes
CLASSES = ["Hello", "Thanks", "Yes", "No", "Help", "Water", "Food"]
```

Then re-run Step 1 (collect) and Step 2 (train).

---

## ❓ Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open webcam` | Check camera is connected; try `VideoCapture(1)` instead of `0` |
| Low accuracy (<80%) | Collect more images (200+) with varied hand positions |
| Prediction is jittery | Increase `SMOOTHING_FRAMES` in `app.py` (try 15–20) |
| No sound on ENTER | Check system audio; on Linux install `espeak`: `sudo apt install espeak` |
| `Model not found` error | Run `train_model.py` before `app.py` |

---

## 🌟 Portfolio Value

- ✅ Real-time computer vision
- ✅ Custom trained neural network
- ✅ Text-to-speech integration
- ✅ Practical accessibility application
- ✅ Impressive live demo on video

---

## 👨‍💻 Author

**Ali Muddassar** — CS Student · Aspiring Data Scientist  
[GitHub](https://github.com/Ali-Muddassar) · [LinkedIn](https://www.linkedin.com/in/ali-muddassar-17466a3b7)
