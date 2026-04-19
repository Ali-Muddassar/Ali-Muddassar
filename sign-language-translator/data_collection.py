"""
Step 1 — Data Collection
========================
Run this script to collect hand-sign images for each letter/word class.

Usage:
    python data_collection.py

Controls:
    Press  's'  to START saving frames for the current class.
    Press  'q'  to move to the NEXT class (or quit after the last one).

Output:
    data/
      A/  (100 images)
      B/  (100 images)
      ...
"""

import cv2
import os
import mediapipe as mp

# ── Configuration ────────────────────────────────────────────────────────────
CLASSES        = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DEL", "NOTHING"]
IMAGES_PER_CLASS = 100
DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
# ─────────────────────────────────────────────────────────────────────────────

mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils


def collect():
    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Connect a camera and try again.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:

        for class_name in CLASSES:
            class_dir = os.path.join(DATA_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)

            print(f"\n[INFO] Get ready for class: '{class_name}'")
            print("       Press 's' to start capturing, 'q' to skip/quit.")

            # ── Wait for user to press 's' ───────────────────────────────────
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                cv2.putText(
                    frame,
                    f"Class: {class_name}  |  Press 's' to start",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    break
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("[INFO] Collection stopped by user.")
                    return

            # ── Capture frames ───────────────────────────────────────────────
            count = 0
            while count < IMAGES_PER_CLASS:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = hands.process(rgb)
                if results.multi_hand_landmarks:
                    for hand_lm in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_lm, mp_hands.HAND_CONNECTIONS
                        )

                cv2.putText(
                    frame,
                    f"Class: {class_name}  [{count}/{IMAGES_PER_CLASS}]",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 200, 255),
                    2,
                )
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(1)

                img_path = os.path.join(class_dir, f"{count:04d}.jpg")
                cv2.imwrite(img_path, frame)
                count += 1

            print(f"[INFO] Saved {count} images for '{class_name}'.")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Data collection complete!  Run train_model.py next.")


if __name__ == "__main__":
    collect()
