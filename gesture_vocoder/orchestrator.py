import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time
import sys

from gesture_vocoder.ipc.shared_state import SharedVocoderState
from gesture_vocoder.vision.pipeline import vision_process
from gesture_vocoder.audio.engine import audio_process

def main():
    # 1. Enforce 'spawn' start method to prevent CoreAudio/ASIO corruption
    mp.set_start_method("spawn", force=True)
    
    print("[Orchestrator] Starting Air Vocoder...")
    
    # 2. Create Shared Memory Block
    shm_size = ctypes.sizeof(SharedVocoderState)
    try:
        shm = shared_memory.SharedMemory(create=True, size=shm_size)
        print(f"[Orchestrator] Created shared memory: {shm.name} ({shm_size} bytes)")
    except Exception as e:
        print(f"[Orchestrator] Failed to create shared memory: {e}")
        sys.exit(1)
        
    # Initialize the struct memory to zeros/defaults
    state = SharedVocoderState.from_buffer(shm.buf)
    state.is_running = True
    state.has_gesture = False
    state.active_chord_id = 0
    state.confidence = 0.0
    state.num_notes = 0
    
    # 3. Create a stop event
    stop_event = mp.Event()
    
    # 4. Spawn processes
    vision_p = mp.Process(target=vision_process, args=(shm.name, stop_event), name="VisionProcess")
    audio_p = mp.Process(target=audio_process, args=(shm.name, stop_event), name="AudioProcess")
    
    vision_p.start()
    audio_p.start()
    
    print("[Orchestrator] Processes spawned. Press Ctrl+C in this terminal to exit.")
    
    # 5. Monitor loop
    try:
        while vision_p.is_alive() and audio_p.is_alive():
            time.sleep(0.5)
            # If the vision process signals quit (Q pressed), we exit
            if not state.is_running:
                print("[Orchestrator] Quit signal received from Vision process.")
                break
    except KeyboardInterrupt:
        print("\n[Orchestrator] Keyboard interrupt detected.")
    
    # 6. Graceful Shutdown
    print("[Orchestrator] Shutting down...")
    stop_event.set()
    state.is_running = False
    
    # Give them a moment to exit cleanly
    vision_p.join(timeout=3.0)
    audio_p.join(timeout=3.0)
    
    if vision_p.is_alive():
        print("[Orchestrator] Terminating Vision process...")
        vision_p.terminate()
    if audio_p.is_alive():
        print("[Orchestrator] Terminating Audio process...")
        audio_p.terminate()
        
    del state # Release pointer
    shm.close()
    shm.unlink()
    print("[Orchestrator] Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()
