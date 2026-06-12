import ctypes

class SharedVocoderState(ctypes.Structure):
    """
    C-struct mapped over shared memory to guarantee lock-free 
    communication between the vision process and the ASIO audio callback.
    """
    _fields_ = [
        ("is_running", ctypes.c_bool),
        ("has_gesture", ctypes.c_bool),
        ("chord_id", ctypes.c_int32),
        ("confidence", ctypes.c_float),
        ("num_notes", ctypes.c_int32),
        ("midi_notes", ctypes.c_int32 * 6),  # Max 6 notes for chords
    ]
