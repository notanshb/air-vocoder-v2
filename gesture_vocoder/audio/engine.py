import time
import numpy as np
import sounddevice as sd
from multiprocessing import shared_memory
import pedalboard
from typing import Optional

from gesture_vocoder.ipc.shared_state import SharedVocoderState

class PolyphonicSynth:
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.phases = np.zeros(6, dtype=np.float32)
    
    def generate(self, midi_notes: np.ndarray, num_notes: int, num_frames: int) -> np.ndarray:
        """Generates a polyphonic sawtooth wave for the given MIDI notes."""
        out = np.zeros(num_frames, dtype=np.float32)
        if num_notes == 0:
            return out
            
        t = np.arange(num_frames, dtype=np.float32) / self.sample_rate
        
        for i in range(num_notes):
            freq = 440.0 * (2.0 ** ((midi_notes[i] - 69) / 12.0))
            # Sawtooth wave generator
            phase_inc = freq * 2.0 * np.pi / self.sample_rate
            phase = self.phases[i] + t * freq * 2.0 * np.pi
            
            # Simple sawtooth approximation using modulo
            # Sawtooth: 2 * (t * freq - floor(t * freq + 0.5))
            saw = 2.0 * ( (phase / (2*np.pi)) - np.floor((phase / (2*np.pi)) + 0.5) )
            out += saw
            
            self.phases[i] = (self.phases[i] + num_frames * phase_inc) % (2.0 * np.pi)
            
        return out / max(1, num_notes) # Normalize

def audio_process(shm_name: str, stop_event):
    """
    Main loop for the audio process. Runs isolated.
    Reads from shared memory lock-free and generates audio via ASIO/default stream.
    """
    print("[Audio] Connecting to shared memory...")
    shm = shared_memory.SharedMemory(name=shm_name)
    state = SharedVocoderState.from_buffer(shm.buf)
    
    sample_rate = 44100.0
    block_size = 512
    
    synth = PolyphonicSynth(sample_rate)
    
    # We will use Pedalboard for a lush Chorus/Reverb effect on the synth output
    board = pedalboard.Pedalboard([
        pedalboard.Chorus(rate_hz=1.0, depth=0.25, centre_delay_ms=7.0, feedback=0.0, mix=0.5),
        pedalboard.Reverb(room_size=0.6, damping=0.5, wet_level=0.3, dry_level=0.8)
    ])
    
    def audio_callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"[Audio Status] {status}")
            
        # 1. Non-blocking read from shared memory
        has_gesture = state.has_gesture
        num_notes = state.num_notes
        
        if not has_gesture or num_notes == 0:
            outdata.fill(0)
            return
            
        # Extract notes into a numpy array (cheap copy)
        notes = np.array(state.midi_notes[:num_notes], dtype=np.int32)
        
        # 2. Synthesize carrier wave (Sawtooth)
        carrier = synth.generate(notes, num_notes, frames)
        
        # 3. Vocoder / Ring Modulation
        # Multiply the carrier point-by-point with the absolute microphone input.
        # This instantly tracks syllables without "spurts", creating a classic vocoder envelope.
        mic_mono = np.mean(indata, axis=1) # mix to mono
        
        # Optional noise gate: if it's too quiet, mute entirely
        rms = np.sqrt(np.mean(mic_mono**2))
        if rms < 0.005:
            modulated = np.zeros_like(carrier)
        else:
            # Multiply by absolute value for amplitude envelope, boost gain
            modulated = carrier * np.abs(mic_mono) * 8.0
        
        # 4. Process through Pedalboard (FX)
        # Pedalboard expects shape (channels, frames)
        modulated_2d = np.vstack((modulated, modulated)) # Stereo
        
        try:
            # Note: plugin() allocates numpy arrays, but for small block_size it's acceptable here
            fx_out = board(modulated_2d, sample_rate, reset=False)
            # Write to output (transpose back to frames, channels)
            outdata[:] = fx_out.T
        except Exception as e:
            # Fallback if pedalboard fails
            outdata[:, 0] = modulated
            outdata[:, 1] = modulated

    print(f"[Audio] Starting SoundDevice stream...")
    # Open the stream using default devices
    try:
        stream = sd.Stream(
            samplerate=sample_rate,
            blocksize=block_size,
            channels=2,
            callback=audio_callback,
            latency='low'
        )
        with stream:
            while not stop_event.is_set() and state.is_running:
                time.sleep(0.1)
    except Exception as e:
        print(f"[Audio] Stream error: {e}")
    finally:
        del state # Release the pointer
        shm.close()
        print("[Audio] Process exited cleanly.")
