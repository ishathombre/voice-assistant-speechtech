"""
simple_asr_whisper.py

The simplest version of the ASR step, with faster-whisper only:
1. Record from the microphone until you stop talking.
2. Turn the recording into text with faster-whisper (on this computer).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # avoids an Anaconda crash on Windows, keep first

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
SAMPLE_RATE = 16000        # samples per second, the standard for speech recognition
BLOCK_SECONDS = 0.1        # the microphone is read in pieces of 0.1 s
SILENCE_LEVEL = 0.01       # pieces quieter than this count as silence
STOP_AFTER_SILENCE = 1.0   # stop recording after this many seconds of silence
MAX_SECONDS = 15           # never record longer than this


# ------------------------------------------------------------------
# Step 1: live recording
# ------------------------------------------------------------------
def record(path="recording.wav"):
    """Record from the microphone until the speaker stops talking, and save it as a WAV file."""
    block_size = round(SAMPLE_RATE * BLOCK_SECONDS)   # 1600 samples = 0.1 s
    blocks = []               # every piece of audio recorded so far
    quiet_blocks = 0          # how many quiet pieces in a row
    started_talking = False   # has the speaker said anything yet?

    print("Listening... speak now.")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as mic:
        while True:
            block, overflowed = mic.read(block_size)   # wait for the next 0.1 s of sound
            blocks.append(block)
            loudness = np.abs(block).mean()            # how loud this piece is

            if loudness > SILENCE_LEVEL:
                started_talking = True
                quiet_blocks = 0
            else:
                quiet_blocks += 1

            if started_talking and quiet_blocks * BLOCK_SECONDS >= STOP_AFTER_SILENCE:
                break                                  # the speaker has finished
            if len(blocks) * BLOCK_SECONDS >= MAX_SECONDS:
                break                                  # safety limit reached
    print("Recording finished.")

    audio = np.concatenate(blocks)                     # join the pieces into one recording
    sf.write(path, audio, SAMPLE_RATE)                 # save it as a WAV file
    return path


# ------------------------------------------------------------------
# Step 2: faster-whisper (runs on this computer, free)
# ------------------------------------------------------------------
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")


def transcribe_local(path):
    """Turn a recording into text with faster-whisper."""
    audio, sample_rate = sf.read(path, dtype="float32")   # read the WAV file into numbers
    segments, info = whisper_model.transcribe(audio)
    return " ".join(segment.text.strip() for segment in segments)


# ------------------------------------------------------------------
# Run it: record, then transcribe
# ------------------------------------------------------------------
recording = record()
print("faster-whisper:", transcribe_local(recording))
