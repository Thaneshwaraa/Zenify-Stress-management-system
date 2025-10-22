import os
import csv
import time
import threading
import json
import random
from datetime import datetime
from collections import deque

import bz2
import numpy as np
import cv2
import dlib
from imutils import face_utils
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# Fix PyTorch 2.6+ compatibility for HSEmotion
try:
    import torch
    # Patch torch.load to allow older model files
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except:
    pass

# Try to import HSEmotion for better accuracy
try:
    from hsemotion.facial_emotions import HSEmotionRecognizer
    HSEMOTION_AVAILABLE = True
except ImportError:
    HSEMOTION_AVAILABLE = False
    print("HSEmotion not available. Install with: pip install hsemotion")

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(BASE_DIR, "Code")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RECOMMENDATIONS_FILE = os.path.join(BASE_DIR, "recommendations.json")
os.makedirs(LOG_DIR, exist_ok=True)

MODEL_NAME = "_mini_XCEPTION.102-0.66.hdf5"
MODEL_PATHS = [
    os.path.join(CODE_DIR, MODEL_NAME),
    os.path.join(BASE_DIR, MODEL_NAME),
]

PREDICTOR_DAT = os.path.join(CODE_DIR, "shape_predictor_68_face_landmarks.dat")
PREDICTOR_BZ2 = os.path.join(CODE_DIR, "shape_predictor_68_face_landmarks.dat.bz2")

# ---------- Helpers ----------

def resolve_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def ensure_shape_predictor(dat_path, bz2_path):
    if os.path.exists(dat_path):
        return dat_path
    if os.path.exists(bz2_path):
        with bz2.open(bz2_path, 'rb') as f_in, open(dat_path, 'wb') as f_out:
            f_out.write(f_in.read())
        return dat_path
    raise FileNotFoundError("shape predictor not found")


# ---------- Models ----------
MODEL_PATH = resolve_first_existing(MODEL_PATHS)
if MODEL_PATH is None:
    raise FileNotFoundError(f"Could not find '{MODEL_NAME}' in {MODEL_PATHS}")

PREDICTOR_PATH = ensure_shape_predictor(PREDICTOR_DAT, PREDICTOR_BZ2)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
emotion_classifier = load_model(MODEL_PATH, compile=False)

# Initialize HSEmotion if available (better accuracy)
hsemotion_model = None
if HSEMOTION_AVAILABLE:
    try:
        hsemotion_model = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device='cpu')
        print("✅ HSEmotion model loaded successfully - using high-accuracy emotion detection (85%+)")
    except Exception as e:
        print(f"❌ Failed to load HSEmotion: {e}")
        print("⚠️  Falling back to mini-XCEPTION model (66% accuracy)")
        HSEMOTION_AVAILABLE = False

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
# HSEmotion uses: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
HSEMOTION_TO_STANDARD = {
    'Anger': 'angry',
    'Contempt': 'disgust',
    'Disgust': 'disgust',
    'Fear': 'scared',
    'Happiness': 'happy',
    'Neutral': 'neutral',
    'Sadness': 'sad',
    'Surprise': 'surprised'
}

AGE_GROUP_EXERCISES = {
    "teen": [
        "30s energized shake-out: gently shake arms and legs to release pent-up energy.",
        "30s mindful listening: close eyes and sync breaths to a calming song intro.",
        "30s seated spinal twist on each side to loosen the mid-back.",
        "30s gratitude pause: name one win from today and smile intentionally."
    ],
    "young-adult": [
        "30s box breathing with upright posture and relaxed shoulders.",
        "30s desk stretch: roll shoulders and stretch wrists slowly.",
        "30s figure-eight eye movement while keeping your neck relaxed.",
        "30s visualization: picture a calm place and breathe into the scene."
    ],
    "adult": [
        "30s 4-7-8 breathing cycle repeated gently.",
        "30s progressive release: tense shoulders for 5s, release, repeat.",
        "30s neck mobility: slow half-circle rolls right and left.",
        "30s mindful walk-in-place focusing on heel-to-toe steps."
    ],
    "senior": [
        "30s diaphragmatic breathing with hands on abdomen for feedback.",
        "30s gentle seated ankle circles to boost circulation.",
        "30s palm press and release to relax hands and wrists.",
        "30s serene visualization of a favorite peaceful memory."
    ]
}

AGE_GROUP_LABELS = {
    "teen": "13-17 years",
    "young-adult": "18-35 years",
    "adult": "36-55 years",
    "senior": "56+ years"
}

AGE_GROUP_QUESTIONS = {
    "teen": [
        "Have you felt overwhelmed by school assignments recently?",
        "Do you find it hard to fall asleep because you are worrying about upcoming tasks?",
        "Have you noticed your heart racing when thinking about exams or grades?",
        "Have you skipped hobbies you enjoy because you felt stressed or tired?",
        "Do you often feel tense when interacting with classmates or friends?",
        "Have you felt pressure to meet expectations from family or teachers this week?",
        "Do you get frequent headaches or stomachaches during busy school days?",
        "Have you been irritable or frustrated without knowing why?",
        "Do you check your phone late at night because of school-related notifications?",
        "Do you feel that you do not have enough downtime to relax each day?"
    ],
    "young-adult": [
        "Have you worried about work or studies immediately after waking up this week?",
        "Do you feel your workload has been unmanageable over the past few days?",
        "Have you skipped meals because of a tight schedule recently?",
        "Do you experience tension in your neck or shoulders after work or classes?",
        "Have you felt anxious about financial responsibilities lately?",
        "Do you find it hard to disconnect from screens at night?",
        "Have you felt guilty about not giving enough time to friends or family?",
        "Do you experience restlessness or agitation during your downtime?",
        "Have you struggled to focus because of racing thoughts this week?",
        "Do you feel you lack control over your daily planning?"
    ],
    "adult": [
        "Have you felt pulled between work and personal responsibilities this week?",
        "Do you worry about providing for your family or dependents frequently?",
        "Have you experienced persistent muscle tension or soreness lately?",
        "Do you struggle to unwind even after the workday ends?",
        "Have you had trouble staying asleep due to stress?",
        "Do you feel pressed for time to care for your own wellbeing?",
        "Have you noticed irritability impacting family interactions?",
        "Do you feel mentally drained before the day is over?",
        "Have you skipped planned breaks or meals because of obligations?",
        "Do you find it hard to enjoy hobbies because your mind stays on tasks?"
    ],
    "senior": [
        "Have you worried about health concerns more than usual this week?",
        "Do you feel anxious about maintaining your independence?",
        "Have you experienced restlessness or trouble relaxing lately?",
        "Do you feel tense when schedules or routines change unexpectedly?",
        "Have you found it hard to concentrate on reading or hobbies?",
        "Do you feel lonely or isolated during the day?",
        "Have you noticed physical signs of stress such as tight muscles?",
        "Do you worry about keeping up with family responsibilities or caregiving?",
        "Have you had difficulty getting restful sleep recently?",
        "Do you feel overwhelmed by managing appointments or finances?"
    ]
}

STRESS_LEVEL_MESSAGES = {
    "low": "Your answers suggest a low stress impact right now. Keep up the helpful routines that support your calm.",
    "medium": "Your responses show moderate stress. Try short breaks, focused breathing, or light movement to reset.",
    "high": "Your answers indicate elevated stress. Consider taking a guided exercise now and reaching out for support if you need it."
}

STRESS_TOOLKIT = {
    "high": [
        {
            "title": "Ground & Center",
            "icon": "🧘",
            "summary": "Interrupt the stress spike and signal safety to your body.",
            "actions": [
                "Plant your feet, breathe in for 4 seconds and out for 6 while naming 3 things you can see.",
                "Press your palms together for 5 seconds, release, and repeat to feel connected to the moment.",
                "Anchor with the 5-4-3-2-1 senses scan to let your brain know you are safe right now."
            ]
        },
        {
            "title": "Release Tension",
            "icon": "💪",
            "summary": "Let the body lead the mind by relaxing overworked muscles.",
            "actions": [
                "Roll shoulders back, then forward, and let them drop on an exhale.",
                "Relax your jaw by gently opening and closing it while humming softly.",
                "Stretch wrists and neck slowly, syncing movement with long exhales."
            ]
        },
        {
            "title": "Reach & Reset",
            "icon": "🤝",
            "summary": "Get perspective and reassurance from supportive voices.",
            "actions": [
                "Send a quick message to someone you trust sharing one feeling and one need.",
                "Schedule a short check-in with a friend or mentor for later today.",
                "Choose one task you can postpone or delegate to reduce immediate load."
            ]
        }
    ],
    "medium": [
        {
            "title": "Micro Break Ritual",
            "icon": "⏱️",
            "summary": "Balance your nervous system with structured pauses.",
            "actions": [
                "Stand up, stretch, and walk for 90 seconds focusing on heel-to-toe steps.",
                "Sip water slowly, noticing the temperature and texture with each swallow.",
                "Reset posture: lift ribs, soften shoulders, take three steady breaths."
            ]
        },
        {
            "title": "Mind Stack",
            "icon": "🧠",
            "summary": "Tame racing thoughts by capturing and organizing them.",
            "actions": [
                "Write one worry, one fact, and one next best action in your notes app.",
                "Decide a mini-win for the next 20 minutes and set a timer.",
                "Move lower-priority tasks to a later block to free mental bandwidth."
            ]
        },
        {
            "title": "Positive Distraction",
            "icon": "🎧",
            "summary": "Shift mood with uplifting input for a few minutes.",
            "actions": [
                "Play a favorite upbeat track and breathe in rhythm for the chorus.",
                "Watch a 2-minute laugh clip or motivational short from the media grid.",
                "Read one encouraging message you keep for days like this."
            ]
        }
    ],
    "low": [
        {
            "title": "Protect the Calm",
            "icon": "🛡️",
            "summary": "Keep helpful habits intact so your stress stays low.",
            "actions": [
                "Schedule your next recovery pause now and treat it as non-negotiable.",
                "Do a 60-second gratitude jot in a journal or notes app.",
                "Stretch or walk lightly to maintain circulation and alertness."
            ]
        },
        {
            "title": "Future Buffer",
            "icon": "📅",
            "summary": "Reduce tomorrow's stress by clearing small friction points today.",
            "actions": [
                "Batch quick replies or tasks into a single focused 15-minute window.",
                "Prep tomorrow's top priority and set out any materials you need.",
                "Block off a no-meeting micro oasis on your calendar."
            ]
        },
        {
            "title": "Connection Boost",
            "icon": "💬",
            "summary": "Share the good energy to reinforce your resilience.",
            "actions": [
                "Send a thank-you or encouragement note to someone in your circle.",
                "Plan a light social moment—even a virtual coffee—to stay supported.",
                "Reflect on one personal strength that kept stress low today."
            ]
        }
    ]
}

# ---------- Stress computation ----------
# Keep bounded history to avoid unbounded growth
points_history = deque(maxlen=300)  # ~10s if 30 FPS
blink_history = deque(maxlen=150)  # Track blinks over ~5s
jaw_history = deque(maxlen=300)  # Track jaw tension
mouth_history = deque(maxlen=300)  # Track mouth aspect ratio
history_lock = threading.Lock()
last_log_time = 0.0

# Exponential moving average for smoothing
ema_stress = 0.0
ema_alpha = 0.25  # Balanced for responsiveness and stability

# Emotion smoothing to prevent flickering
emotion_history = deque(maxlen=15)  # Track last 15 emotion detections (~1.5 seconds)
emotion_lock = threading.Lock()

# Performance optimization: cache computed values
cached_stress_data = {
    'score': 0.0,
    'level': 'low',
    'emotion': 'neutral',
    'label': 'not stressed',
    'last_update': 0.0,
    'initialized': False,
    'confidence': 0.0,  # Detection confidence score
    'components': {
        'emotion': 0.0,
        'eyebrow': 0.0,
        'blink': 0.0,
        'jaw': 0.0,
        'mouth': 0.0
    }
}
cache_lock = threading.Lock()


def eyebrow_distance(leye, reye):
    return float(np.linalg.norm(leye - reye))


def eye_aspect_ratio(eye):
    """
    Calculate Eye Aspect Ratio (EAR) to detect blinks.
    Lower EAR = eye more closed (blink or squint)
    """
    # Compute vertical eye distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Compute horizontal eye distance
    C = np.linalg.norm(eye[0] - eye[3])
    # EAR formula
    ear = (A + B) / (2.0 * C + 1e-6)
    return float(ear)


def mouth_aspect_ratio(mouth):
    """
    Calculate Mouth Aspect Ratio (MAR) to detect mouth tension.
    Higher MAR = mouth more open
    Lower MAR = mouth tense/closed (stress indicator)
    """
    # Vertical distances
    A = np.linalg.norm(mouth[2] - mouth[10])  # 51-59
    B = np.linalg.norm(mouth[4] - mouth[8])   # 53-57
    # Horizontal distance
    C = np.linalg.norm(mouth[0] - mouth[6])   # 49-55
    mar = (A + B) / (2.0 * C + 1e-6)
    return float(mar)


def jaw_tension_score(jaw_points):
    """
    Calculate jaw tension based on jaw line curvature.
    More angular/tense jaw = higher stress
    """
    # Calculate the angle at the jaw corners
    left_jaw = jaw_points[0:5]
    right_jaw = jaw_points[-5:]
    
    # Measure the "sharpness" of jaw angle (more acute = more tension)
    left_angle_point = left_jaw[2]
    right_angle_point = right_jaw[2]
    chin_point = jaw_points[len(jaw_points)//2]
    
    # Calculate vectors
    left_vec = left_angle_point - chin_point
    right_vec = right_angle_point - chin_point
    
    # Angle between vectors (smaller angle = more tension)
    cos_angle = np.dot(left_vec, right_vec) / (np.linalg.norm(left_vec) * np.linalg.norm(right_vec) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    # Normalize: smaller angle (more acute) = higher tension
    # Typical relaxed jaw angle is around 120-140 degrees (2.1-2.4 rad)
    # Tense jaw angle is around 90-110 degrees (1.6-1.9 rad)
    tension = float(np.clip(1.0 - (angle / np.pi), 0.0, 1.0))
    return tension


def detect_blink_rate(ear_history):
    """
    Improved blink rate detection from EAR history.
    Rapid blinking (>25 blinks/min) indicates stress.
    Normal rate is 12-20 blinks/min.
    """
    if len(ear_history) < 45:  # Need at least 1.5 seconds of data for reliability
        return 0.0
    
    # Detect blinks: EAR drops below threshold
    EAR_THRESHOLD = 0.21  # Calibrated threshold
    blinks = 0
    in_blink = False
    frames_closed = 0
    frames_open = 0
    
    for ear in ear_history:
        if ear < EAR_THRESHOLD:
            frames_closed += 1
            frames_open = 0
            # Count as blink if eyes closed for 2-8 frames (realistic blink: 66-266ms at 30fps)
            if not in_blink and 2 <= frames_closed <= 8:
                blinks += 1
                in_blink = True
        else:
            frames_open += 1
            frames_closed = 0
            # Reset blink state after eyes open for at least 2 frames
            if in_blink and frames_open >= 2:
                in_blink = False
    
    # Calculate blinks per minute (assuming 30 FPS)
    time_window_seconds = len(ear_history) / 30.0
    blinks_per_minute = (blinks / time_window_seconds) * 60.0
    
    # Improved normalization curve based on research
    # Normal: 12-20 bpm, Stress: 25+ bpm, Very stressed: 35+ bpm
    if blinks_per_minute < 8:
        stress = 0.10  # Very low (possibly tired or drowsy)
    elif blinks_per_minute < 12:
        stress = 0.15  # Low-normal
    elif blinks_per_minute < 20:
        stress = 0.20  # Normal baseline (not stressed)
    elif blinks_per_minute < 25:
        stress = 0.40  # Slightly elevated
    elif blinks_per_minute < 30:
        stress = 0.60  # Moderate stress
    elif blinks_per_minute < 35:
        stress = 0.75  # Elevated stress
    elif blinks_per_minute < 40:
        stress = 0.85  # High stress
    else:
        stress = 0.95  # Very high stress
    
    return float(stress)


def normalize_values(points_series, current_disp):
    # Use robust stats and relative deviation to reduce sensitivity to distance/scale
    if len(points_series) < 5:  # Reduced from 10 for faster response
        return 0.0, "low"
    series = np.array(points_series, dtype=np.float32)
    median = float(np.median(series))
    mad = float(np.median(np.abs(series - median))) + 1e-6  # robust scale

    # Relative deviation (higher -> more change -> potentially higher stress)
    # Made more sensitive by reducing the divisor
    rel_dev = abs(current_disp - median) / (mad * 4.0)  # Reduced from 6.0 to 4.0 for higher sensitivity
    rel_dev = float(np.clip(rel_dev, 0.0, 1.0))

    # Apply a slight amplification curve to make eyebrow changes more impactful
    stress = rel_dev ** 0.8  # Slight amplification of smaller changes

    # Levels after smoothing will be decided later; here keep simple thresholds
    if stress >= 0.6:  # Updated thresholds to match main logic
        level = "high"
    elif stress >= 0.3:
        level = "medium"
    else:
        level = "low"
    return stress, level


def emotion_finder(face_rect, gray_frame, color_frame=None):
    """
    Enhanced emotion detection using HSEmotion (if available) or fallback to mini-XCEPTION.
    HSEmotion provides significantly better accuracy (85%+ vs 66%).
    Includes preprocessing improvements and confidence filtering.
    """
    x, y, w, h = face_utils.rect_to_bb(face_rect)
    x, y = max(0, x), max(0, y)
    
    # Try HSEmotion first (better accuracy)
    if HSEMOTION_AVAILABLE and hsemotion_model is not None and color_frame is not None:
        try:
            # Extract face ROI with padding for better context (15% padding)
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.15)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(color_frame.shape[1], x + w + pad_x)
            y2 = min(color_frame.shape[0], y + h + pad_y)
            
            face_roi = color_frame[y1:y2, x1:x2]
            
            if face_roi.size > 0 and face_roi.shape[0] > 30 and face_roi.shape[1] > 30:
                # Enhance image quality for better detection
                # Convert to RGB for processing
                face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                
                # Resize to optimal size for HSEmotion (224x224 works best)
                face_roi_resized = cv2.resize(face_roi_rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
                
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better lighting
                lab = cv2.cvtColor(face_roi_resized, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                face_roi_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
                
                # Apply slight Gaussian blur to reduce noise
                face_roi_processed = cv2.GaussianBlur(face_roi_enhanced, (3, 3), 0)
                
                # Convert back to BGR for HSEmotion
                face_roi_final = cv2.cvtColor(face_roi_processed, cv2.COLOR_RGB2BGR)
                
                # Get emotion predictions with confidence scores
                emotion, scores = hsemotion_model.predict_emotions(face_roi_final, logits=False)
                
                # Get the emotion with highest confidence
                if isinstance(emotion, list) and len(emotion) > 0:
                    detected_emotion = emotion[0]
                    # Get confidence score if available
                    if isinstance(scores, (list, np.ndarray)) and len(scores) > 0:
                        confidence = float(np.max(scores[0]) if isinstance(scores[0], np.ndarray) else scores[0])
                    else:
                        confidence = 1.0
                else:
                    detected_emotion = emotion
                    confidence = 1.0
                
                # Only accept detection if confidence is reasonable (> 0.25)
                if confidence > 0.25:
                    # Map to standard emotion set
                    standard_emotion = HSEMOTION_TO_STANDARD.get(detected_emotion, 'neutral')
                    return standard_emotion
                    
        except Exception as e:
            # Fall back to mini-XCEPTION if HSEmotion fails
            pass
    
    # Fallback to mini-XCEPTION model with improved preprocessing
    roi = gray_frame[y:y+h, x:x+w]
    if roi.size == 0:
        return "neutral"
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi = clahe.apply(roi)
    
    # Resize with better interpolation
    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_CUBIC)
    
    # Normalize
    roi = roi.astype("float32") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    
    # Get predictions
    preds = emotion_classifier.predict(roi, verbose=0)[0]
    
    # Only accept if confidence is high enough (> 0.35)
    max_confidence = float(np.max(preds))
    if max_confidence > 0.35:
        return EMOTIONS[int(np.argmax(preds))]
    else:
        return "neutral"  # Default to neutral if uncertain


def smooth_emotion(new_emotion):
    """
    Smooth emotion detection to prevent flickering.
    Uses majority voting over recent detections with temporal weighting.
    """
    with emotion_lock:
        emotion_history.append(new_emotion)
        
        # Need at least 5 samples for smoothing
        if len(emotion_history) < 5:
            return new_emotion
        
        # Count occurrences with temporal weighting (recent emotions weighted more)
        emotion_counts = {}
        total_weight = 0.0
        
        for i, emotion in enumerate(emotion_history):
            # Weight increases linearly (older = 1.0, newest = 2.0)
            weight = 1.0 + (i / max(1, len(emotion_history) - 1))
            emotion_counts[emotion] = emotion_counts.get(emotion, 0.0) + weight
            total_weight += weight
        
        # Normalize counts
        for emotion in emotion_counts:
            emotion_counts[emotion] /= total_weight
        
        # Return the emotion with highest weighted count
        return max(emotion_counts.items(), key=lambda x: x[1])[0]


def calculate_confidence(history_lengths):
    """
    Calculate detection confidence based on available data.
    More historical data = higher confidence in the assessment.
    Returns confidence score 0.0-1.0
    """
    # Check how much data we have for each feature
    points_len, blink_len, jaw_len, mouth_len = history_lengths
    
    # Minimum required for each feature (increased for better reliability)
    min_required = {
        'points': 60,    # ~2 seconds for stable baseline
        'blink': 90,     # ~3 seconds for blink rate detection
        'jaw': 60,       # ~2 seconds for jaw tension
        'mouth': 60      # ~2 seconds for mouth tension
    }
    
    # Optimal data amounts for maximum confidence
    optimal = {
        'points': 150,   # ~5 seconds
        'blink': 150,    # ~5 seconds
        'jaw': 150,      # ~5 seconds
        'mouth': 150     # ~5 seconds
    }
    
    # Calculate confidence for each feature with gradual increase
    def calc_feature_conf(length, min_req, opt):
        if length < min_req:
            # Below minimum: confidence scales linearly from 0 to 0.6
            return (length / min_req) * 0.6
        else:
            # Above minimum: confidence scales from 0.6 to 1.0
            excess = min(length - min_req, opt - min_req)
            return 0.6 + (excess / (opt - min_req)) * 0.4
    
    points_conf = calc_feature_conf(points_len, min_required['points'], optimal['points'])
    blink_conf = calc_feature_conf(blink_len, min_required['blink'], optimal['blink'])
    jaw_conf = calc_feature_conf(jaw_len, min_required['jaw'], optimal['jaw'])
    mouth_conf = calc_feature_conf(mouth_len, min_required['mouth'], optimal['mouth'])
    
    # Overall confidence is the weighted average
    # Blink and eyebrow are more reliable indicators, so weight them higher
    overall_confidence = (
        0.30 * points_conf +   # Eyebrow movement
        0.30 * blink_conf +    # Blink rate
        0.20 * jaw_conf +      # Jaw tension
        0.20 * mouth_conf      # Mouth tension
    )
    
    return float(overall_confidence)


def get_emotion_stress_score(emotion):
    """
    Convert emotion to stress score with heavy emphasis on negative emotions.
    Positive emotions = very low stress, Negative emotions = very high stress
    """
    emotion_stress_map = {
        # Highly positive emotions (very low stress)
        "happy": 0.05,
        "surprised": 0.15,
        
        # Neutral emotion (low-moderate stress)
        "neutral": 0.25,
        
        # Negative emotions (very high stress) - INCREASED VALUES
        "sad": 0.85,
        "angry": 0.95,
        "scared": 0.98,
        "disgust": 0.90
    }
    
    return emotion_stress_map.get(emotion, 0.4)  # Default to moderate if unknown


# ---------- Video generator ----------
from flask import Flask, render_template, Response, jsonify, request, make_response, url_for

app = Flask(__name__)

# Simple in-memory store for completed questionnaires (non-persistent)
age_questionnaire_results = []

# CSV logging
CSV_PATH = os.path.join(LOG_DIR, "sessions.csv")
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "stress_score", "level", "emotion"])  # header


class CaptureManager:
    """Background camera capture to avoid stream on/off flicker."""
    def __init__(self):
        self.cap = None
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        self.latest = None
        self.last_ok = 0.0
        self.selected_camera = None  # User-selected camera (idx, backend)
        self.available_cameras = []  # List of detected cameras
        # Prefer MediaFoundation on Windows; avoids some DirectShow issues
        # Scan more camera indices to detect USB webcams (0-5)
        self.trials = [
            (0, cv2.CAP_MSMF), (1, cv2.CAP_MSMF), (2, cv2.CAP_MSMF), 
            (3, cv2.CAP_MSMF), (4, cv2.CAP_MSMF), (5, cv2.CAP_MSMF),
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
        ]
        self._detect_cameras()
        self._load_camera_config()
    
    def _detect_cameras(self):
        """Detect all available cameras at startup."""
        print("🔍 Detecting available cameras...")
        self.available_cameras = []
        for idx, backend in self.trials:
            try:
                c = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
                if c is not None and c.isOpened():
                    ret, frame = c.read()
                    if ret and frame is not None:
                        backend_name = "CAP_MSMF" if backend == cv2.CAP_MSMF else "Default"
                        width = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        self.available_cameras.append({
                            'index': idx,
                            'backend': backend,
                            'backend_name': backend_name,
                            'resolution': f"{width}x{height}"
                        })
                        print(f"  ✅ Camera {len(self.available_cameras)}: Index {idx} ({backend_name}) - {width}x{height}")
                    c.release()
            except Exception:
                pass
        
        if not self.available_cameras:
            print("  ⚠️  No cameras detected!")
        else:
            print(f"  📷 Total cameras found: {len(self.available_cameras)}")
    
    def _load_camera_config(self):
        """Load camera selection from config file if it exists."""
        config_file = os.path.join(BASE_DIR, "camera_config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    camera_idx = config.get('camera_index')
                    backend_name = config.get('backend', 'CAP_MSMF')
                    
                    # Convert backend name to constant
                    backend = cv2.CAP_MSMF if backend_name == 'CAP_MSMF' else 0
                    
                    # Find this camera in available cameras
                    for i, cam in enumerate(self.available_cameras):
                        if cam['index'] == camera_idx and cam['backend'] == backend:
                            self.selected_camera = (camera_idx, backend)
                            print(f"  📌 Using saved camera preference: Index {camera_idx} ({backend_name})")
                            return
                    
                    # If not found in available cameras, still try to use it
                    self.selected_camera = (camera_idx, backend)
                    print(f"  📌 Attempting to use saved camera: Index {camera_idx} ({backend_name})")
            except Exception as e:
                print(f"  ⚠️  Failed to load camera config: {e}")
    
    def set_camera(self, camera_index):
        """Set which camera to use (0 = first detected, 1 = second, etc.)"""
        if 0 <= camera_index < len(self.available_cameras):
            cam = self.available_cameras[camera_index]
            self.selected_camera = (cam['index'], cam['backend'])
            print(f"📷 Switched to camera {camera_index + 1}: Index {cam['index']} ({cam['backend_name']})")
            # Restart capture with new camera
            if self.running:
                self._reopen_camera()
            return True
        return False
    
    def _reopen_camera(self):
        """Close current camera and reopen."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _open(self):
        # If user selected a specific camera, try that first
        trials_to_use = []
        if self.selected_camera is not None:
            trials_to_use = [self.selected_camera]
        else:
            # Otherwise try all detected cameras, then fall back to full scan
            trials_to_use = [(cam['index'], cam['backend']) for cam in self.available_cameras]
            if not trials_to_use:
                trials_to_use = self.trials
        
        # Try backends/indices
        for idx, backend in trials_to_use:
            try:
                backend_name = "CAP_MSMF" if backend == cv2.CAP_MSMF else "Default"
                c = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
                if c is None or not c.isOpened():
                    if c is not None:
                        c.release()
                    continue
                # Optimize camera settings for performance and low latency
                c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                c.set(cv2.CAP_PROP_FPS, 30)  # 30 FPS is optimal for balance
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
                c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPEG for better performance
                # Some webcams/drivers misbehave with forced exposure settings; keep defaults
                # c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                # c.set(cv2.CAP_PROP_EXPOSURE, -5)
                self.cap = c
                print(f"✅ Camera opened successfully: Index {idx}, Backend: {backend_name}")
                return True
            except Exception as e:
                print(f"❌ Failed to open camera at index {idx} with {backend_name}: {e}")
                try:
                    if c is not None:
                        c.release()
                except Exception:
                    pass
        print("⚠️  No camera found after trying all available options")
        return False

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

    def _loop(self):
        backoff = 0.1
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if not self._open():
                    time.sleep(min(2.0, backoff))
                    backoff = min(2.0, backoff * 1.5)
                    continue
                backoff = 0.1
            ok, frame = self.cap.read()
            if not ok or frame is None:
                # Trigger reopen on next iteration
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(0.05)
                continue
            with self.lock:
                self.latest = frame
                self.last_ok = time.time()
            # Reduced sleep for better responsiveness
            time.sleep(1.0 / 30.0)  # Target 30 FPS capture (optimal balance)

    def get_frame(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()


# Start capture manager
cap_manager = CaptureManager()
cap_manager.start()


def log_row(stress_score, level, emotion):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow().isoformat(), f"{stress_score:.3f}", level, emotion])


# Background capture manager replaces manual open_camera/locks


def gen_frames():
    # If camera is not ready, show a static message until background thread warms up
    warmup_deadline = time.time() + 5.0
    while cap_manager.get_frame() is None:
        # Avoid waiting forever; show a static frame and let the loop try
        if time.time() > warmup_deadline:
            break
        img = np.zeros((300, 500, 3), dtype=np.uint8)
        cv2.putText(img, "Waiting for camera...", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        ret, buffer = cv2.imencode('.jpg', img)
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.25)

    last_good_frame = None
    idle_counter = 0
    frame_skip_counter = 0
    last_emit = time.time()
    min_frame_interval = 1.0 / 25.0  # Target 25 FPS output (smooth but not laggy)

    while True:
        frame = cap_manager.get_frame()
        if frame is None:
            # If capture thread lost the device, sleep briefly but keep serving last good
            idle_counter += 1
            if last_good_frame is not None and (idle_counter % 5 == 0):
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
                ret, buffer = cv2.imencode('.jpg', last_good_frame, encode_params)
                if ret:
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)
            continue
        idle_counter = 0
        
        # Skip frames if we're emitting too fast (prevent browser buffer buildup)
        current_time = time.time()
        if current_time - last_emit < min_frame_interval:
            time.sleep(0.01)  # Small sleep to prevent CPU spinning
            continue

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Skip expensive processing every few frames for better performance
        current_time = time.time()
        process_frame = (current_time - cached_stress_data['last_update'] > 0.066)  # Process every 66ms (~15 FPS analysis)
        
        if process_frame:
            detections = detector(gray, 0)  # No upsampling for better performance
            
            if detections:
                d = detections[0]  # Process only first face for performance
                
                # Get emotion classification first (primary indicator)
                # Pass color frame for better accuracy with HSEmotion
                raw_emotion = emotion_finder(d, gray, frame)
                
                # Apply emotion smoothing to prevent flickering
                emotion = smooth_emotion(raw_emotion)
                
                # Calculate all facial features for comprehensive stress analysis
                shape = predictor(gray, d)
                shape = face_utils.shape_to_np(shape)
                
                # Extract facial landmarks
                (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
                (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
                (lEyeBegin, lEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                (rEyeBegin, rEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
                (mouthBegin, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
                (jawBegin, jawEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
                
                leyebrow = shape[lBegin:lEnd]
                reyebrow = shape[rBegin:rEnd]
                left_eye = shape[lEyeBegin:lEyeEnd]
                right_eye = shape[rEyeBegin:rEyeEnd]
                mouth = shape[mouthBegin:mouthEnd]
                jaw = shape[jawBegin:jawEnd]

                # Draw facial features for visualization (optimized)
                # Use simpler drawing for better performance
                cv2.polylines(frame, [cv2.convexHull(leyebrow)], True, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.polylines(frame, [cv2.convexHull(reyebrow)], True, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.polylines(frame, [cv2.convexHull(left_eye)], True, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.polylines(frame, [cv2.convexHull(right_eye)], True, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.polylines(frame, [cv2.convexHull(mouth)], True, (255, 0, 0), 1, cv2.LINE_AA)

                # Calculate eyebrow-based stress
                x, y, w, h = face_utils.rect_to_bb(d)
                face_width = max(1.0, float(w))
                distq = eyebrow_distance(leyebrow[-1], reyebrow[0]) / face_width
                
                # Calculate eye aspect ratios for blink detection
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Calculate mouth aspect ratio for tension detection
                mar = mouth_aspect_ratio(mouth)
                
                # Calculate jaw tension
                jaw_tension = jaw_tension_score(jaw)

                with history_lock:
                    # Update all feature histories
                    points_history.append(distq)
                    blink_history.append(avg_ear)
                    mouth_history.append(mar)
                    jaw_history.append(jaw_tension)
                    
                    # Calculate individual stress indicators
                    eyebrow_stress, _ = normalize_values(points_history, distq)
                    blink_stress = detect_blink_rate(blink_history)
                    
                    # Mouth tension: Detect tightness/compression (stress indicator)
                    # Lower MAR than baseline = tense/compressed mouth = stress
                    if len(mouth_history) >= 30:  # Need more data for reliable baseline
                        mouth_array = np.array(list(mouth_history))
                        mouth_baseline = float(np.percentile(mouth_array, 50))  # Median as baseline
                        mouth_std = float(np.std(mouth_array)) + 1e-6
                        
                        # Detect compression (MAR below baseline indicates tension)
                        deviation = (mouth_baseline - mar) / mouth_std
                        # Positive deviation = compressed/tense, negative = relaxed/open
                        mouth_stress = float(np.clip(deviation * 0.5, 0.0, 1.0))
                    else:
                        mouth_stress = 0.0
                    
                    # Jaw tension: Use baseline comparison for better accuracy
                    if len(jaw_history) >= 30:  # Need sufficient history
                        jaw_array = np.array(list(jaw_history))
                        jaw_baseline = float(np.percentile(jaw_array, 25))  # Lower quartile as relaxed baseline
                        jaw_current = jaw_tension
                        
                        # Compare current to baseline (higher = more tense)
                        jaw_stress = float(np.clip((jaw_current - jaw_baseline) / (0.3 + 1e-6), 0.0, 1.0))
                    else:
                        jaw_stress = 0.0
                    
                    # **ENHANCED WEIGHTED STRESS CALCULATION**
                    # Improved weighting for better accuracy
                    # Reduced emotion weight, increased physiological indicators
                    # Emotion: 25% (important but not dominant - can be misleading)
                    # Eyebrow: 20% (facial expression - reliable)
                    # Blink rate: 25% (physiological indicator - very reliable)
                    # Jaw tension: 20% (physical stress - reliable)
                    # Mouth tension: 10% (subtle indicator)
                    emotion_stress = get_emotion_stress_score(emotion)
                    
                    # Calculate physiological stress (non-emotion indicators)
                    physiological_stress = (
                        0.27 * eyebrow_stress +   # 20/75 normalized
                        0.33 * blink_stress +     # 25/75 normalized
                        0.27 * jaw_stress +       # 20/75 normalized
                        0.13 * mouth_stress       # 10/75 normalized
                    )
                    
                    # Combined stress with balanced weighting
                    combined_stress = (
                        0.25 * emotion_stress +
                        0.20 * eyebrow_stress +
                        0.25 * blink_stress +
                        0.20 * jaw_stress +
                        0.10 * mouth_stress
                    )
                    
                    # Smooth combined stress score to avoid jitter
                    global ema_stress
                    if not cached_stress_data['initialized']:
                        # Initialize EMA with first reading for faster startup
                        ema_stress = combined_stress
                    else:
                        ema_stress = (1 - ema_alpha) * ema_stress + ema_alpha * combined_stress
                    stress_score = float(ema_stress)
                    
                    # **IMPROVED: Smart emotion-based adjustment**
                    # Trust physiological indicators more, use emotion as modifier
                    if emotion in ["happy", "surprised"]:
                        # Positive emotions: reduce stress but trust physiological signals
                        if physiological_stress > 0.6:
                            # High physiological stress despite positive emotion (forced smile/surprise)
                            # Trust the body more than the face
                            adjustment = 0.0  # No reduction
                        elif physiological_stress > 0.4:
                            # Moderate physiological stress - small reduction
                            adjustment = -0.08
                        else:
                            # Low physiological stress - genuine positive emotion
                            adjustment = -0.18
                        stress_score = max(0.0, stress_score + adjustment)
                    elif emotion in ["angry", "scared", "disgust"]:
                        # Strong negative emotions: significant stress boost
                        # These emotions are strong stress indicators
                        if physiological_stress > 0.5:
                            # Both emotion and body agree - high stress
                            adjustment = 0.25
                        else:
                            # Emotion shows stress but body is calm - moderate boost
                            adjustment = 0.15
                        stress_score = min(1.0, stress_score + adjustment)
                    elif emotion == "sad":
                        # Sad emotion: moderate stress boost
                        if physiological_stress > 0.4:
                            adjustment = 0.18
                        else:
                            adjustment = 0.10
                        stress_score = min(1.0, stress_score + adjustment)
                    # Neutral: no adjustment, trust the multi-feature analysis
                    
                    # Determine stress level with refined thresholds
                    # These thresholds are calibrated for the 5-feature system
                    if stress_score >= 0.55:  # High stress threshold
                        stress_level = "high"
                        stress_label = "stressed"
                    elif stress_score >= 0.30:  # Medium stress threshold
                        stress_level = "medium"
                        stress_label = "moderately stressed"
                    else:
                        stress_level = "low"
                        stress_label = "not stressed"
                    
                    # Calculate detection confidence
                    confidence = calculate_confidence((
                        len(points_history),
                        len(blink_history),
                        len(jaw_history),
                        len(mouth_history)
                    ))
                
                # Update cache with all component scores
                with cache_lock:
                    cached_stress_data.update({
                        'score': stress_score,
                        'level': stress_level,
                        'emotion': emotion,
                        'label': stress_label,
                        'last_update': current_time,
                        'initialized': True,
                        'confidence': confidence,
                        'components': {
                            'emotion': emotion_stress,
                            'eyebrow': eyebrow_stress,
                            'blink': blink_stress,
                            'jaw': jaw_stress,
                            'mouth': mouth_stress
                        }
                    })
            else:
                # Use cached values when no face detected
                with cache_lock:
                    stress_score = cached_stress_data['score']
                    stress_level = cached_stress_data['level']
                    emotion = cached_stress_data['emotion']
                    stress_label = cached_stress_data['label']
        else:
            # Use cached values when not processing
            with cache_lock:
                stress_score = cached_stress_data['score']
                stress_level = cached_stress_data['level']
                emotion = cached_stress_data['emotion']
                stress_label = cached_stress_data['label']

        # overlay text with emotion-weighted emphasis and debug info
        color = (0, 200, 0) if stress_level == "low" else ((0, 165, 255) if stress_level == "medium" else (0, 0, 255))
        
        # Get confidence for display
        with cache_lock:
            confidence = cached_stress_data.get('confidence', 0.0)
            components = cached_stress_data.get('components', {})
        
        # Main stress display with confidence indicator
        cv2.putText(frame, f"Stress: {int(stress_score*100)}% ({stress_level})", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Emotion: {emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show confidence with color coding (red=low, yellow=medium, green=high)
        conf_color = (0, 255, 0) if confidence > 0.8 else ((0, 255, 255) if confidence > 0.5 else (0, 100, 255))
        cv2.putText(frame, f"Confidence: {int(confidence*100)}%", (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 2)
        
        # Add enhanced debug info showing all stress components
        if components:
            # Display individual component scores with better formatting
            cv2.putText(frame, f"E:{int(components.get('emotion', 0)*100)}% B:{int(components.get('eyebrow', 0)*100)}% Bl:{int(components.get('blink', 0)*100)}% J:{int(components.get('jaw', 0)*100)}% M:{int(components.get('mouth', 0)*100)}%", 
                       (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # draw a border reflecting state
        if stress_level == "high":
            border = (0, 0, 255)
        elif stress_level == "medium":
            border = (0, 165, 255)
        else:
            border = (0, 200, 0)
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), border, 3)

        # Encode JPEG with optimized quality/performance balance
        # Use lower quality for better performance and reduced bandwidth
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, 65,  # Lower quality for better performance
            cv2.IMWRITE_JPEG_OPTIMIZE, 1,   # Enable JPEG optimization
            cv2.IMWRITE_JPEG_PROGRESSIVE, 1  # Progressive JPEG for faster loading
        ]
        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        last_good_frame = frame
        last_emit = time.time()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Mark cache as initialized once we have a valid reading, log once per second
        global last_log_time
        if not cached_stress_data['initialized']:
            with cache_lock:
                cached_stress_data['initialized'] = True
        now = time.time()
        if now - last_log_time >= 1.0:
            log_row(stress_score, stress_level, emotion)
            last_log_time = now


@app.route('/')
def index():
    # Home page
    return render_template('home.html')


@app.route('/camera-select')
def camera_select_page():
    # Camera selection page
    return render_template('camera_select.html')


@app.route('/session')
def session_page():
    # 30s live session page with age-group context
    age_group = request.args.get('age_group') or request.cookies.get('age_group') or 'adult'
    if age_group not in AGE_GROUP_LABELS:
        age_group = 'adult'
    resp = make_response(render_template('session.html', age_group=age_group, age_label=AGE_GROUP_LABELS[age_group]))
    resp.set_cookie('age_group', age_group, max_age=60 * 60 * 24 * 7)
    return resp


@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire_page():
    age_group = request.args.get('age_group') or request.cookies.get('age_group') or 'adult'
    if age_group not in AGE_GROUP_QUESTIONS:
        age_group = 'adult'
    questions = AGE_GROUP_QUESTIONS[age_group]
    age_label = AGE_GROUP_LABELS.get(age_group, AGE_GROUP_LABELS['adult'])

    if request.method == 'POST':
        responses = []
        for idx in range(1, len(questions) + 1):
            responses.append(request.form.get(f'responses_{idx}', 'no'))
        yes_count = sum(1 for r in responses if r == 'yes')
        score_percentage = int((yes_count / max(len(questions), 1)) * 100)

        if score_percentage >= 70:
            stress_level = 'high'
        elif score_percentage >= 40:
            stress_level = 'medium'
        else:
            stress_level = 'low'

        result_summary = {
            'age_group': age_group,
            'age_label': age_label,
            'total_questions': len(questions),
            'yes_count': yes_count,
            'score_percentage': score_percentage,
            'stress_level': stress_level,
        }
        age_questionnaire_results.append(result_summary)

        messages = {
            'high': 'Consider starting a guided exercise right away and speak with someone you trust.',
            'medium': 'Take a short break, hydrate, and try the breathing exercise to reset.',
            'low': 'Keep up your healthy routines and check in again soon.'
        }

        return render_template(
            'questionnaire_result.html',
            summary=result_summary,
            message=STRESS_LEVEL_MESSAGES[stress_level],
            recommendation=messages[stress_level]
        )

    questions_payload = []
    for idx, question in enumerate(questions, start=1):
        questions_payload.append({'index': idx, 'text': question})

    return render_template(
        'questionnaire.html',
        age_group=age_group,
        age_label=age_label,
        questions=questions_payload
    )


@app.route('/result')
def result_page():
    # Read level/score from query (fallback to current status)
    level = request.args.get('level')
    try:
        score = float(request.args.get('score')) if request.args.get('score') is not None else None
    except Exception:
        score = None
    # Use smoothed score if missing
    if level is None or score is None:
        with history_lock:
            global ema_stress
            score = float(ema_stress)
            if score >= 0.7:
                level = 'high'
            elif score >= 0.4:
                level = 'medium'
            else:
                level = 'low'
    tips = recommendations(level)
    
    # Get comprehensive recommendations for the result page
    stress_recs = get_stress_relief_recommendations(level)
    # Map the data structure to match template expectations
    mapped_recommendations = {
        'comedy': stress_recs.get('comedy_videos', []),
        'music': stress_recs.get('relaxing_music', []),
        'motivation': stress_recs.get('motivational_content', [])
    }
    
    return render_template('result.html', level=level, score=score, tips=tips, recommendations=mapped_recommendations)


@app.route('/exercise')
def exercise_page():
    # Always base exercises on smoothed latest level with updated thresholds
    age_group = request.args.get('age_group') or request.cookies.get('age_group') or 'adult'
    if age_group not in AGE_GROUP_EXERCISES:
        age_group = 'adult'
    with history_lock:
        s = float(ema_stress)
    if s >= 0.6:  # Updated threshold
        lvl = 'high'
    elif s >= 0.3:  # Updated threshold
        lvl = 'medium'
    else:
        lvl = 'low'
    tips = recommendations(lvl)
    toolkit = STRESS_TOOLKIT.get(lvl, [])
    exercises = AGE_GROUP_EXERCISES.get(age_group, AGE_GROUP_EXERCISES['adult'])
    age_label = AGE_GROUP_LABELS.get(age_group, AGE_GROUP_LABELS['adult'])

    full_recommendations = get_stress_relief_recommendations(lvl)
    recommendations_by_category = {
        'comedy': full_recommendations.get('comedy_videos', []),
        'music': full_recommendations.get('relaxing_music', []),
        'motivation': full_recommendations.get('motivational_content', [])
    }

    def enrich_with_thumbnails(items):
        enriched = []
        for item in items:
            if isinstance(item, dict):
                enriched.append({
                    'title': item.get('title'),
                    'url': item.get('url'),
                    'thumbnail': item.get('thumbnail'),
                    'artist': item.get('artist'),
                    'speaker': item.get('speaker'),
                    'duration': item.get('duration')
                })
        return enriched

    recommendations_by_category['comedy'] = enrich_with_thumbnails(recommendations_by_category['comedy'])
    recommendations_by_category['music'] = enrich_with_thumbnails(recommendations_by_category['music'])
    recommendations_by_category['motivation'] = enrich_with_thumbnails(recommendations_by_category['motivation'])

    # Provide a default timer in seconds (front-end can control)
    return render_template(
        'exercise.html',
        level=lvl,
        tips=tips,
        score=s,
        exercises=exercises,
        age_label=age_label,
        age_group=age_group,
        media_recommendations=recommendations_by_category,
        stress_toolkit=toolkit
    )


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    # Provide current recommendation based on smoothed score with updated thresholds
    with cache_lock:
        score = cached_stress_data.get('score', 0.0)
        level = cached_stress_data.get('level', 'low')
        emotion = cached_stress_data.get('emotion', 'neutral')
        confidence = cached_stress_data.get('confidence', 0.0)
        components = cached_stress_data.get('components', {})
        initialized = cached_stress_data.get('initialized', False)
    
    tips = recommendations(level)
    return jsonify({
        "level": level,
        "tips": tips,
        "score": score,
        "emotion": emotion,
        "confidence": confidence,
        "components": components,
        "initialized": initialized
    })


@app.route('/cameras')
def list_cameras():
    """List all available cameras."""
    cameras = []
    for i, cam in enumerate(cap_manager.available_cameras):
        cameras.append({
            'id': i,
            'name': f"Camera {i + 1} (Index {cam['index']})",
            'index': cam['index'],
            'backend': cam['backend_name'],
            'resolution': cam['resolution']
        })
    return jsonify({
        'cameras': cameras,
        'total': len(cameras)
    })


@app.route('/camera/select/<int:camera_id>', methods=['POST'])
def select_camera(camera_id):
    """Select which camera to use."""
    if cap_manager.set_camera(camera_id):
        return jsonify({
            'success': True,
            'message': f'Switched to camera {camera_id + 1}'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Invalid camera ID'
        }), 400


def load_recommendations():
    """Load recommendations from JSON file."""
    try:
        with open(RECOMMENDATIONS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Shuffle each category to randomize order
            for key in ["comedy_videos", "relaxing_music", "motivational_content"]:
                if key in data and isinstance(data[key], list):
                    random.shuffle(data[key])
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback data if file doesn't exist or is invalid
        return {
            "comedy_videos": [],
            "relaxing_music": [],
            "motivational_content": []
        }


def get_stress_relief_recommendations(stress_level, session_duration_minutes=5):
    """Get personalized recommendations based on stress level and session duration."""
    data = load_recommendations()
    recommendations = {
        "comedy_videos": [],
        "relaxing_music": [],
        "motivational_content": [],
        "breathing_tips": []
    }
    
    # Get breathing tips based on stress level
    if stress_level == 'high':
        recommendations["breathing_tips"] = [
            "Box breathing: 4s inhale, 4s hold, 4s exhale, 4s hold × 4 cycles",
            "Progressive muscle relaxation: tense/release shoulders, jaw, hands",
            "Grounding 5-4-3-2-1: senses scan for 2 minutes"
        ]
        # For high stress, prioritize calming content - Fixed filtering
        calming_music = [item for item in data.get("relaxing_music", []) if item.get("mood") in ["calming", "peaceful", "soothing", "serene", "tranquil"]]
        if calming_music:
            recommendations["relaxing_music"] = random.sample(calming_music, min(3, len(calming_music)))
        
        high_rated_comedy = [item for item in data.get("comedy_videos", []) if item.get("stress_relief_rating", 0) >= 8]
        if high_rated_comedy:
            recommendations["comedy_videos"] = random.sample(high_rated_comedy, min(2, len(high_rated_comedy)))
        else:
            # Fallback to any comedy videos
            all_comedy = data.get("comedy_videos", [])
            if all_comedy:
                recommendations["comedy_videos"] = random.sample(all_comedy, min(2, len(all_comedy)))
        
    elif stress_level == 'medium':
        recommendations["breathing_tips"] = [
            "Guided breathing: 4s inhale, 6s exhale × 2 minutes",
            "Short walk/stretch for 2 minutes",
            "Reframe: write 1 worry, 1 action next"
        ]
        # For medium stress, mix of uplifting and calming - Fixed filtering
        mixed_music = [item for item in data.get("relaxing_music", []) if item.get("mood") in ["uplifting", "peaceful", "gentle", "calm"]]
        if mixed_music:
            recommendations["relaxing_music"] = random.sample(mixed_music, min(2, len(mixed_music)))
        else:
            # Fallback to any music
            all_music = data.get("relaxing_music", [])
            if all_music:
                recommendations["relaxing_music"] = random.sample(all_music, min(2, len(all_music)))
        
        all_comedy = data.get("comedy_videos", [])
        if all_comedy:
            recommendations["comedy_videos"] = random.sample(all_comedy, min(3, len(all_comedy)))
        
        all_motivational = data.get("motivational_content", [])
        if all_motivational:
            recommendations["motivational_content"] = random.sample(all_motivational, min(1, len(all_motivational)))
        
    else:  # low stress
        recommendations["breathing_tips"] = [
            "Maintain: 4s inhale, 4s exhale × 1 minute",
            "Posture check and shoulder roll",
            "Gratitude note: 1 thing going well"
        ]
        # For low stress, focus on maintaining positive mood - Fixed filtering
        all_comedy = data.get("comedy_videos", [])
        if all_comedy:
            recommendations["comedy_videos"] = random.sample(all_comedy, min(2, len(all_comedy)))
        
        # Try to find uplifting music, fallback to any music
        uplifting_music = [item for item in data.get("relaxing_music", []) if "uplifting" in item.get("mood", "").lower()]
        if uplifting_music:
            recommendations["relaxing_music"] = random.sample(uplifting_music, min(1, len(uplifting_music)))
        else:
            all_music = data.get("relaxing_music", [])
            if all_music:
                recommendations["relaxing_music"] = random.sample(all_music, min(1, len(all_music)))
    
    return recommendations


def recommendations(level: str):
    """Generate breathing/relaxation tips based on stress level (legacy function)."""
    recs = get_stress_relief_recommendations(level)
    return recs["breathing_tips"]


@app.route('/start-breathing', methods=['POST'])
def start_breathing():
    # For extensibility; front-end handles the timer
    return jsonify({"ok": True})


@app.route('/session-end-recommendations')
def session_end_recommendations():
    """Get comprehensive recommendations after a stress monitoring session (API)."""
    # Get current stress level and session duration
    with history_lock:
        score = float(ema_stress)
        if score >= 0.6:
            level = 'high'
        elif score >= 0.3:
            level = 'medium'
        else:
            level = 'low'
    
    # Get session duration from query params (in minutes), default to 5
    session_duration = request.args.get('duration', 5, type=int)
    
    # Get comprehensive recommendations
    recs = get_stress_relief_recommendations(level, session_duration)
    
    return jsonify({
        "session_summary": {
            "stress_level": level,
            "stress_score": int(score * 100),
            "session_duration_minutes": session_duration
        },
        "recommendations": recs,
        "message": get_session_end_message(level)
    })


@app.route('/session-complete')
def session_complete():
    """Display session end page with recommendations."""
    # Get current stress level and session duration
    with history_lock:
        score = float(ema_stress)
        if score >= 0.6:
            level = 'high'
        elif score >= 0.3:
            level = 'medium'
        else:
            level = 'low'
    
    # Get session duration from query params (in minutes), default to 5
    session_duration = request.args.get('duration', 5, type=int)
    
    # Get comprehensive recommendations
    recs = get_stress_relief_recommendations(level, session_duration)
    
    # Debug: Print what we got
    print(f"DEBUG - Stress Level: {level}")
    print(f"DEBUG - Comedy Videos Count: {len(recs.get('comedy_videos', []))}")
    print(f"DEBUG - Relaxing Music Count: {len(recs.get('relaxing_music', []))}")
    print(f"DEBUG - Motivational Content Count: {len(recs.get('motivational_content', []))}")
    
    session_summary = {
        "stress_level": level,
        "stress_score": int(score * 100),
        "session_duration_minutes": session_duration
    }
    
    return render_template('session_end.html', 
                         session_summary=session_summary,
                         recommendations=recs,
                         message=get_session_end_message(level))


@app.route('/debug-recommendations')
def debug_recommendations():
    """Debug endpoint to test recommendations loading."""
    data = load_recommendations()
    return jsonify({
        "comedy_videos_count": len(data.get("comedy_videos", [])),
        "relaxing_music_count": len(data.get("relaxing_music", [])),
        "motivational_content_count": len(data.get("motivational_content", [])),
        "sample_comedy": data.get("comedy_videos", [])[:2] if data.get("comedy_videos") else [],
        "sample_music": data.get("relaxing_music", [])[:2] if data.get("relaxing_music") else []
    })


def get_session_end_message(level):
    """Get encouraging message based on stress level."""
    messages = {
        "high": "You've been through a stressful period. Take some time for yourself with these calming activities.",
        "medium": "You're doing well managing your stress. Here are some activities to help you feel even better.",
        "low": "Great job staying relaxed! Here are some positive activities to keep your mood up."
    }
    return messages.get(level, "Session complete! Here are some recommendations for you.")


import atexit
import signal

def _shutdown(*_args, **_kwargs):
    try:
        cap_manager.stop()
    except Exception:
        pass

# Stop camera only when the process exits, not after each request
atexit.register(_shutdown)
try:
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
except Exception:
    pass


if __name__ == '__main__':
    # Run: python app.py, then open http://localhost:5000
    # use_reloader=False prevents double-spawn that can clash with camera
    # threaded=True keeps the MJPEG stream responsive while handling other requests
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)