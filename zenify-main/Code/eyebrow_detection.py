import os
import bz2
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import load_model


def resolve_first_existing(paths):
    """Return the first path that exists from a list of candidates."""
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def ensure_shape_predictor(dat_path, bz2_path):
    """Ensure the uncompressed dlib shape predictor exists; extract from .bz2 if needed."""
    if os.path.exists(dat_path):
        return dat_path
    if os.path.exists(bz2_path):
        try:
            with bz2.open(bz2_path, 'rb') as f_in, open(dat_path, 'wb') as f_out:
                f_out.write(f_in.read())
            return dat_path
        except Exception as e:
            raise RuntimeError(f"Failed to extract '{bz2_path}' -> '{dat_path}': {e}")
    raise FileNotFoundError(
        f"Could not find shape predictor. Expected either '{dat_path}' or '{bz2_path}'."
    )


def eye_brow_distance(leye, reye):
    global points
    distq = dist.euclidean(leye, reye)
    points.append(float(distq))
    return distq


def emotion_finder(face_rect, gray_frame):
    global emotion_classifier
    EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

    # Crop face ROI from grayscale frame
    x, y, w, h = face_utils.rect_to_bb(face_rect)
    x, y = max(0, x), max(0, y)
    roi_frame = gray_frame[y : y + h, x : x + w]
    if roi_frame.size == 0:
        return "not stressed"

    roi = cv2.resize(roi_frame, (64, 64))
    roi = roi.astype("float32") / 255.0
    roi = img_to_array(roi)  # shape (64,64,1)
    roi = np.expand_dims(roi, axis=0)

    preds = emotion_classifier.predict(roi, verbose=0)[0]
    label = EMOTIONS[preds.argmax()]

    # Map emotions to stressed/not stressed
    if label in ["scared", "sad", "angry", "disgust"]:
        return "stressed"
    return "not stressed"


def normalize_values(points_series, current_disp):
    """
    Normalize current eyebrow distance against observed min/max and compute a stress score.
    Returns score in [0,1] and a label using a 0.75 threshold.
    Safeguards against division by zero during initial frames.
    """
    # Need enough history to establish a range
    if len(points_series) < 5:
        return 0.0, "low_stress"

    pmin = float(np.min(points_series))
    pmax = float(np.max(points_series))
    denom = pmax - pmin
    if denom <= 1e-6:
        normalized_value = 0.0
    else:
        normalized_value = abs(current_disp - pmin) / denom

    # Higher normalized value -> higher change -> higher stress; map via exp decay
    stress_value = float(np.exp(-(normalized_value)))  # 1.0 (low change) -> ~0.37; 0.0 (no change) -> 1.0

    label = "High Stress" if stress_value >= 0.75 else "low_stress"
    return stress_value, label


# ----------- Setup paths and models -----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Dlib shape predictor paths
PREDICTOR_DAT = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
PREDICTOR_BZ2 = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat.bz2")

# Emotion model paths (try local folder, then repo root)
MODEL_NAME = "_mini_XCEPTION.102-0.66.hdf5"
MODEL_PATH = resolve_first_existing([
    os.path.join(BASE_DIR, MODEL_NAME),
    os.path.join(REPO_ROOT, MODEL_NAME),
])
if MODEL_PATH is None:
    raise FileNotFoundError(
        f"Could not find '{MODEL_NAME}' next to this script or at repo root: '{REPO_ROOT}'."
    )

# Ensure shape predictor is available (auto-extract if only .bz2 is present)
PREDICTOR_PATH = ensure_shape_predictor(PREDICTOR_DAT, PREDICTOR_BZ2)

# Initialize models
_detector = dlib.get_frontal_face_detector()
_predictor = dlib.shape_predictor(PREDICTOR_PATH)
emotion_classifier = load_model(MODEL_PATH, compile=False)

# ----------- Video loop -----------
cap = cv2.VideoCapture(0)
points = []

try:
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=500, height=500)

        # Landmark indices for eyebrows
        (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
        (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

        # Preprocess image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = _detector(gray, 0)
        for detection in detections:
            emotion = emotion_finder(detection, gray)
            cv2.putText(
                frame,
                emotion,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if emotion == "not stressed" else (0, 0, 255),
                2,
            )

            shape = _predictor(gray, detection)
            shape = face_utils.shape_to_np(shape)

            leyebrow = shape[lBegin:lEnd]
            reyebrow = shape[rBegin:rEnd]

            # Draw eyebrow contours
            reyebrowhull = cv2.convexHull(reyebrow)
            leyebrowhull = cv2.convexHull(leyebrow)
            cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)

            # Distance between inner eyebrow points
            distq = eye_brow_distance(leyebrow[-1], reyebrow[0])
            stress_value, stress_label = normalize_values(points, distq)

            cv2.putText(
                frame,
                f"stress level: {int(stress_value * 100)}% ({stress_label})",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0) if stress_label == "low_stress" else (0, 0, 255),
                2,
            )

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
finally:
    cv2.destroyAllWindows()
    cap.release()

# Plot collected eyebrow distances
if points:
    plt.plot(range(len(points)), points, "ro")
    plt.title("Eyebrow Distance (proxy for Stress)")
    plt.xlabel("Frame")
    plt.ylabel("Distance (px)")
    plt.show()