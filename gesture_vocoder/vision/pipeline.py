import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass
from typing import Optional
from collections import Counter
import urllib.request, os
from multiprocessing import shared_memory
import ctypes

from gesture_vocoder.ipc.shared_state import SharedVocoderState
from gesture_vocoder.mapping.chord_map import get_gesture_id, get_chord_notes

MODEL_PATH = "models/gesture_recognizer.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)

def ensure_model_exists():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe gesture model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

class LM:
    WRIST = 0; THUMB_MCP = 2; THUMB_IP = 3; THUMB_TIP = 4
    INDEX_MCP = 5; INDEX_PIP = 6; INDEX_DIP = 7; INDEX_TIP = 8
    MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
    RING_MCP = 13; RING_PIP = 14; RING_DIP = 15; RING_TIP = 16
    PINKY_MCP = 17; PINKY_PIP = 18; PINKY_DIP = 19; PINKY_TIP = 20

@dataclass
class FingerState:
    thumb: bool; index: bool; middle: bool; ring: bool; pinky: bool

@dataclass
class GestureEvent:
    name: str; source: str; confidence: float

def _lm_array(landmarks) -> np.ndarray: return np.array([[l.x, l.y, l.z] for l in landmarks])
def _angle_at_joint(a, b, c) -> float:
    ba = a - b; bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
def _dist(a, b) -> float: return float(np.linalg.norm(a - b))

def _finger_extended(pts, tip, dip, pip, mcp, threshold=160.0) -> bool:
    return _angle_at_joint(pts[pip], pts[dip], pts[tip]) > threshold
def _thumb_extended(pts, threshold=150.0) -> bool:
    return _angle_at_joint(pts[LM.THUMB_MCP], pts[LM.THUMB_IP], pts[LM.THUMB_TIP]) > threshold

def get_finger_states(pts) -> FingerState:
    return FingerState(
        thumb=_thumb_extended(pts),
        index=_finger_extended(pts, LM.INDEX_TIP, LM.INDEX_DIP, LM.INDEX_PIP, LM.INDEX_MCP),
        middle=_finger_extended(pts, LM.MIDDLE_TIP, LM.MIDDLE_DIP, LM.MIDDLE_PIP, LM.MIDDLE_MCP),
        ring=_finger_extended(pts, LM.RING_TIP, LM.RING_DIP, LM.RING_PIP, LM.RING_MCP),
        pinky=_finger_extended(pts, LM.PINKY_TIP, LM.PINKY_DIP, LM.PINKY_PIP, LM.PINKY_MCP),
    )

# --- Custom Geometries ---
# Old custom geometries commented out
# def _is_rock_and_roll(pts, fs) -> bool: ...
# def _is_gun(pts, fs) -> bool: ...
# def _is_c_curl(pts, fs) -> bool: ...
# def _is_ok_sign(pts, fs) -> bool: ...
# def _is_spock(pts, fs) -> bool: ...
# def classify_custom(pts, fs) -> Optional[GestureEvent]: ...

def classify_count(fs: FingerState) -> Optional[GestureEvent]:
    # 0 to 8 counting gestures
    if not fs.thumb and not fs.index and not fs.middle and not fs.ring and not fs.pinky:
        return GestureEvent("count_0", "custom", 1.0)
    if not fs.thumb and fs.index and not fs.middle and not fs.ring and not fs.pinky:
        return GestureEvent("count_1", "custom", 1.0)
    if not fs.thumb and fs.index and fs.middle and not fs.ring and not fs.pinky:
        return GestureEvent("count_2", "custom", 1.0)
    if not fs.thumb and fs.index and fs.middle and fs.ring and not fs.pinky:
        return GestureEvent("count_3", "custom", 1.0)
    if not fs.thumb and fs.index and fs.middle and fs.ring and fs.pinky:
        return GestureEvent("count_4", "custom", 1.0)
    if fs.thumb and fs.index and fs.middle and fs.ring and fs.pinky:
        return GestureEvent("count_5", "custom", 1.0)
    if fs.thumb and not fs.index and not fs.middle and not fs.ring and not fs.pinky:
        return GestureEvent("count_6", "custom", 1.0)
    if fs.thumb and fs.index and not fs.middle and not fs.ring and not fs.pinky:
        return GestureEvent("count_7", "custom", 1.0)
    if fs.thumb and fs.index and fs.middle and not fs.ring and not fs.pinky:
        return GestureEvent("count_8", "custom", 1.0)
    return None

class GestureSmoother:
    def __init__(self, window: int = 8, threshold: int = 4, max_missing_frames: int = 15):
        self.window = window; self.threshold = threshold; self.max_missing_frames = max_missing_frames
        self._missing_count = 0; self._history = []; self._current = None

    def update(self, event: Optional[GestureEvent]) -> Optional[GestureEvent]:
        if event is None:
            self._missing_count += 1
            if self._missing_count > self.max_missing_frames:
                self._current = None; self._history.clear()
            return self._current
        
        self._missing_count = 0
        self._history.append(event)
        if len(self._history) > self.window: self._history.pop(0)
        
        if self._history:
            names = [e.name for e in self._history]
            most_common_name, count = Counter(names).most_common(1)[0]
            if count >= self.threshold:
                for e in self._history:
                    if e.name == most_common_name:
                        self._current = e
                        break
        return self._current

MP_GESTURE_MAP = {
    "Victory": "victory", "Thumb_Up": "thumbs_up", "Thumb_Down": "thumbs_down",
    "Open_Palm": "open_palm", "Closed_Fist": "closed_fist", 
    "Pointing_Up": "pointing_up", "ILoveYou": "i_love_you"
}

GESTURE_CHORD_MAP = {
    "count_0": "C major", "count_1": "D minor", "count_2": "E minor",
    "count_3": "F major", "count_4": "G major", "count_5": "A minor",
    "count_6": "B dim", "count_7": "Cmaj7", "count_8": "Dmin7",
}

# Old map
# GESTURE_CHORD_MAP = {
#     "closed_fist": "C major", "pointing_up": "D minor", "victory": "E minor",
#     "open_palm": "F major", "rock_and_roll": "G major", "thumbs_up": "A minor",
#     "c_curl": "Bm7b5", "ok_sign": "A major", "gun": "D major",
#     "i_love_you": "E major", "spock": "Bb major"
# }

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12), (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20), (5,9),(9,13),(13,17)
]

def draw_overlay(frame, gesture, pts_px=None):
    h, w = frame.shape[:2]
    if pts_px is not None:
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts_px[a], pts_px[b], (80, 80, 80), 1)
        for pt in pts_px:
            cv2.circle(frame, pt, 4, (200, 200, 200), -1)
    if gesture:
        chord = GESTURE_CHORD_MAP.get(gesture.name, "-")
        cv2.rectangle(frame, (0, h - 80), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, f"{gesture.name}  [{gesture.source}]", (16, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        cv2.putText(frame, chord, (16, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 220, 140), 2)
    else:
        cv2.rectangle(frame, (0, h - 40), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, "No gesture", (16, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

def build_recogniser():
    ensure_model_exists()
    options = mp_vision.GestureRecognizerOptions(
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode = mp_vision.RunningMode.IMAGE,
        num_hands = 1, min_hand_detection_confidence = 0.7,
        min_hand_presence_confidence = 0.7, min_tracking_confidence = 0.6,
    )
    return mp_vision.GestureRecognizer.create_from_options(options)

def write_to_shared_state(state: SharedVocoderState, gesture: Optional[GestureEvent]):
    if gesture:
        state.has_gesture = True
        state.active_chord_id = get_gesture_id(gesture.name)
        state.confidence = gesture.confidence
        
        notes = get_chord_notes(gesture.name)
        state.num_notes = len(notes)
        for i, n in enumerate(notes):
            if i < 6:
                state.midi_notes[i] = n
    else:
        state.has_gesture = False
        state.active_chord_id = 0
        state.num_notes = 0

def vision_process(shm_name: str, stop_event):
    """
    Main loop for the vision process. Runs isolated.
    Connects to the orchestrator's shared memory.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    state = SharedVocoderState.from_buffer(shm.buf)
    
    recogniser = build_recogniser()
    smoother = GestureSmoother(window=8, threshold=4, max_missing_frames=15)
    cap = cv2.VideoCapture(0)
    print("[Vision] Camera started. Press Q to quit.")
    
    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recogniser.recognize(mp_img)
            
            gesture: Optional[GestureEvent] = None
            pts_px = None
            
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                pts = _lm_array(lms)
                pts_px = [(int(l.x * w), int(l.y * h)) for l in lms]
                fs = get_finger_states(pts)
                
                # Removed MediaPipe gesture logic as requested, forcing custom counting gestures
                gesture = classify_count(fs)
                    
            stable = smoother.update(gesture)
            
            # --- IPC Write (Zero-copy, lock-free) ---
            write_to_shared_state(state, stable)
            
            draw_overlay(frame, stable, pts_px)
            cv2.imshow("Air Vocoder - Gesture Recognition", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
                state.is_running = False
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        del state # Release the pointer before closing shared memory to prevent BufferError
        shm.close()
        print("[Vision] Process exited cleanly.")
