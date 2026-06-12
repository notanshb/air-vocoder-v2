from typing import List

# Gestures from Phase 1 mapped to MIDI notes
# Gestures mapped to count 0-8
CHORD_MIDI_MAP = {
    "count_0": [60, 64, 67],           # C major (C4, E4, G4)
    "count_1": [62, 65, 69],           # D minor (D4, F4, A4)
    "count_2": [64, 67, 71],           # E minor (E4, G4, B4)
    "count_3": [65, 69, 72],           # F major (F4, A4, C5)
    "count_4": [67, 71, 74],           # G major (G4, B4, D5)
    "count_5": [69, 72, 76],           # A minor (A4, C5, E5)
    "count_6": [71, 74, 77],           # B dim (B4, D5, F5)
    "count_7": [60, 64, 67, 71],       # Cmaj7 (C4, E4, G4, B4)
    "count_8": [62, 65, 69, 72],       # Dmin7 (D4, F4, A4, C5)
}

# Old gestures commented out as requested
# CHORD_MIDI_MAP_OLD = {
#     "closed_fist": [60, 64, 67],           # C major (C4, E4, G4)
#     "pointing_up": [62, 65, 69],           # D minor (D4, F4, A4)
#     "victory": [64, 67, 71],               # E minor (E4, G4, B4)
#     "open_palm": [65, 69, 72],             # F major (F4, A4, C5)
#     "rock_and_roll": [67, 71, 74],         # G major (G4, B4, D5)
#     "thumbs_up": [69, 72, 76],             # A minor (A4, C5, E5)
#     "c_curl": [59, 62, 65, 69],            # Bm7b5 (B3, D4, F4, A4)
#     "ok_sign": [69, 73, 76],               # A major (A4, Db5, E5)
#     "gun": [62, 66, 69],                   # D major (D4, Gb4, A4)
#     "i_love_you": [64, 68, 71],            # E major (E4, Ab4, B4)
#     "spock": [58, 62, 65]                  # Bb major (Bb3, D4, F4)
# }

# Assign integer IDs to each gesture for passing via ctypes struct
GESTURE_ID_MAP = {name: idx + 1 for idx, name in enumerate(CHORD_MIDI_MAP.keys())}
ID_GESTURE_MAP = {idx: name for name, idx in GESTURE_ID_MAP.items()}


def get_chord_notes(gesture_name: str) -> List[int]:
    """Return the MIDI notes for a given gesture."""
    return CHORD_MIDI_MAP.get(gesture_name, [])

def get_gesture_id(gesture_name: str) -> int:
    """Return the integer ID for a gesture."""
    return GESTURE_ID_MAP.get(gesture_name, 0)
