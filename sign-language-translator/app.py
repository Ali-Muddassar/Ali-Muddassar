"""
Step 3 — Real-Time Sign Language Translator
============================================
Uses your webcam to detect hand signs frame-by-frame,
predicts the letter/word, builds a sentence, and speaks it aloud.

Usage:
    python app.py

Keyboard Controls:
    SPACE  →  add a space between words
    ENTER  →  speak the current sentence (text-to-speech)
    c      →  clear the sentence buffer
    q      →  quit

Requirements:
    model/sign_language_model.h5
    model/label_map.npy
    (both produced by train_model.py)
"""

import os
import time
import threading
import collections

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tensorflow as tf

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "sign_language_model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "model", "label_map.npy")
# ─────────────────────────────────────────────────────────────────────────────

# ── Prediction smoothing ──────────────────────────────────────────────────────
SMOOTHING_FRAMES   = 10      # vote over last N frames
CONFIDENCE_THRESH  = 0.80    # minimum confidence to accept a prediction
HOLD_FRAMES        = 20      # frames a letter must be stable before adding it
# ─────────────────────────────────────────────────────────────────────────────

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ── Text-to-Speech (runs in background thread) ────────────────────────────────
_tts_engine = pyttsx3.init()
_tts_engine.setProperty("rate", 140)

def speak(text: str) -> None:
    """Speak text in a daemon thread so the webcam loop never blocks."""
    def _run():
        _tts_engine.say(text)
        _tts_engine.runAndWait()
    t = threading.Thread(target=_run, daemon=True)
    t.start()
# ─────────────────────────────────────────────────────────────────────────────


def extract_landmarks_live(hand_landmarks) -> np.ndarray:
    """Flatten 21 × (x, y, z) into a (63,) array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    ).flatten()


def draw_ui(
    frame: np.ndarray,
    prediction: str,
    confidence: float,
    sentence: str,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Current prediction
    cv2.putText(
        frame,
        f"Sign: {prediction}  ({confidence * 100:.1f}%)",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 120),
        2,
    )

    # Sentence at bottom
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 80), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
    cv2.putText(
        frame,
        f"Sentence: {sentence}",
        (10, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )

    # Controls hint
    hints = "SPACE=space  ENTER=speak  C=clear  Q=quit"
    cv2.putText(
        frame,
        hints,
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )

    return frame


def run():
    # ── Load model & labels ───────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run train_model.py first to train the model."
        )

    print("[INFO] Loading model …")
    model  = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABEL_PATH, allow_pickle=True)
    print(f"[INFO] Model loaded. Classes: {list(labels)}")

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sentence          = ""
    recent_preds      = collections.deque(maxlen=SMOOTHING_FRAMES)
    stable_label      = ""
    stable_count      = 0
    last_added_label  = ""
    last_added_time   = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            prediction = "—"
            confidence = 0.0

            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                hand_lm = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76),  thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2),
                )

                features = extract_landmarks_live(hand_lm).reshape(1, -1)
                probs    = model.predict(features, verbose=0)[0]
                idx      = int(np.argmax(probs))
                confidence = float(probs[idx])

                if confidence >= CONFIDENCE_THRESH:
                    prediction = labels[idx]
                    recent_preds.append(prediction)

                    # Majority vote over recent frames
                    if len(recent_preds) == SMOOTHING_FRAMES:
                        counter     = collections.Counter(recent_preds)
                        voted_label = counter.most_common(1)[0][0]

                        if voted_label == stable_label:
                            stable_count += 1
                        else:
                            stable_label = voted_label
                            stable_count = 1

                        # Add letter after holding HOLD_FRAMES and cooldown
                        now = time.time()
                        if (
                            stable_count >= HOLD_FRAMES
                            and (
                                voted_label != last_added_label
                                or now - last_added_time > 1.5
                            )
                            and voted_label not in ("NOTHING",)
                        ):
                            if voted_label == "SPACE":
                                sentence += " "
                            elif voted_label == "DEL":
                                sentence = sentence[:-1]
                            else:
                                sentence += voted_label

                            last_added_label = voted_label
                            last_added_time  = now
                            stable_count     = 0
                else:
                    recent_preds.clear()
                    stable_label = ""
                    stable_count = 0

            frame = draw_ui(frame, prediction, confidence, sentence)
            cv2.imshow("🤚 Sign Language Translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 13:                  # ENTER — speak
                if sentence.strip():
                    speak(sentence.strip())
            elif key == ord(" "):            # SPACE — add space
                sentence += " "
            elif key == ord("c"):            # C — clear
                sentence = ""

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Translator closed.")


if __name__ == "__main__":
    run()
