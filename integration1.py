import numpy as np
import asyncio, base64, time
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from hume import AsyncHumeClient
from hume.tts import PostedUtterance, PostedUtteranceVoiceWithName
import pygame

# Voice rec settings
SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.1
SILENCE_LEVEL = 0.01
STOP_AFTER_SILENCE = 1.0
MAX_SECONDS = 15

# Initialize external tools
load_dotenv()
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")

_client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    #stream=True
    )

client = AsyncHumeClient(api_key=os.environ["HUME_API_KEY"])

VOICE = PostedUtteranceVoiceWithName(name="Ava Song", provider="HUME_AI")


def record(path="recording.wav"):
    """Record from the microphone until the speaker stops talking, and save it as a WAV file."""
    block_size = round(SAMPLE_RATE * BLOCK_SECONDS)   # 1600 samples = 0.1 s
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

    audio = np.concatenate(blocks)                     # join the pieces into one recording
    sf.write(path, audio, SAMPLE_RATE)                 # save it as a WAV file
    return path


def transcribe_local(path):
    """Turn a recording into text with faster-whisper."""
    audio, sample_rate = sf.read(path, dtype="float32")   # read the WAV file into numbers
    segments, info = whisper_model.transcribe(audio)
    return " ".join(segment.text.strip() for segment in segments)


def get_LLM_response(prompt: str, memory=None):
    """Sends a prompt to DeepSeek V4.1 Flash and returns (reply, updated_memory)"""

    # Add prompt to the LLM memory
    if not memory:
        print("Memory is empty, starting a new conversation")
        memory = []
        memory.append(
            {"role": "system",
             "content": """
             You are an LLM inside an emotional voice assistant pipeline.
             The goal of the assistant is to say a sentence requested by a user in a given emotion.
             Your role is to take the ASR output and prep it for the TTS by providing instructions on how to act out the voice.
             Please parse the “instructions” and “utterance” and output them in JSON format.

             EXAMPLE INPUT:
             Can you say the following sentence in a cheerful voice: “we ate this assignment.”

             EXAMPLE JSON OUTPUT:
             {
                 "instructions": “Bright and cheerful, with a warm smile in the voice. Light and bouncy pacing, slightly faster than normal, pitch lifting on the important words.”,
                 "utterance": “we ate this assignment
             }
             """
             })

    memory.append({"role": "user", "content": prompt})

    # Send a call to an LLM with the memory
    response = _client.chat.completions.create(
        model="deepseek-flash",
        messages=memory,
        response_format={'type': 'json_object'},
        #reasoning_effort="high/low",
        #extra_body={"thinking": {"type": "disabled"}}
    )

    # Add the LLM's response to the memory
    msg = response.choices[0].message
    memory.append({"role": "assistant", "content": msg.content})

    # Convert json str to a python dict
    tts_input = json.loads(msg.content)

    return tts_input, memory


async def speak(text, instruct, path):
    t0 = time.time()
    result = await client.tts.synthesize_json(
        utterances=[PostedUtterance(text=text, description=instruct, voice=VOICE)],
    )
    secs = time.time() - t0
    with open(path, "wb") as f:
        f.write(base64.b64decode(result.generations[0].audio))
    return secs

def play_audio(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pass

async def main():
    audio_response_path = "sample_audio.mp3"

    record(path="recording.wav")
    transcript = transcribe_local("recording.wav")
    response1, memory = get_LLM_response(transcript)
    await speak(
    response1["utterance"],
    response1["instructions"],
    audio_response_path
    )

    play_audio(audio_response_path)

    return audio_response_path

if __name__ == "__main__":
    asyncio.run(main())
