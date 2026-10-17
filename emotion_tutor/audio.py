import numpy as np
import pygame
import sounddevice as sd
import soundfile as sf

pygame.mixer.init()

SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.1
SILENCE_LEVEL = 0.01
STOP_AFTER_SILENCE = 1.0
MAX_SECONDS = 15

def record(path="recording.wav"):
    """Record from the microphone until the speaker stops talking, and save it as a WAV file."""
    block_size = round(SAMPLE_RATE * BLOCK_SECONDS)
    blocks = []               # every piece of audio recorded so far
    quiet_blocks = 0          # how many quiet pieces in a row
    started_talking = False

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

    audio = np.concatenate(blocks)
    sf.write(path, audio, SAMPLE_RATE)
    return path


def play_audio(path):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

    clock = pygame.time.Clock()

    while pygame.mixer.music.get_busy():
        clock.tick(30)

    pygame.mixer.music.unload()
