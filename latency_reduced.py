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
import re

# Voice rec settings
SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.1
SILENCE_LEVEL = 0.01
STOP_AFTER_SILENCE = 1.0
MAX_SECONDS = 15

# Dev switches
TEXT_MODE = False
PLAY_SOUND = True
# Initialize external tools
load_dotenv()
pygame.mixer.init()

SYSTEM_PROMPT = """
You are the brain of a friendly voice assistant for children, the Emotion Tutor.
Everything you write is spoken aloud by a text-to-speech (TTS) system that can act out
emotions. Talk like a kind teacher: short sentences, simple words, child-friendly.

The user sends a SAY request (a sentence in an emotion), a QUESTION, or both.
Your reply is a list of segments, spoken in order. There are two kinds:
- TEACHER segment: warm teacher voice, gentle pacing.
- ACTED segment: the user's sentence, said exactly as given, in the requested emotion.

BRIDGE LINE: a teacher segment that comes right before an acted segment must END with a
short bridge line that leads into the acting, such as "Now let's get into character!",
"Okay, time to act it out!" or "Here we go, in character!". Vary the wording. Never put a
bridge line in an acted segment or in a segment that isn't followed by one.

- SAY only: a teacher segment that is just a bridge line, then the acted segment.
- QUESTION only: a teacher segment with the answer (at most 3 short sentences, using the
  earlier conversation if relevant) ending with a bridge line, then an acted segment with
  the user's most recent say-sentence in the same emotion, unless they asked for another.
  If there was no earlier say-sentence, give only the answer, with no bridge line.
- BOTH: the answer plus a bridge line, then the acted segment.
- Unclear, empty or nonsense input: one teacher segment kindly asking them to say it
  again, with no bridge line.
- Scary, violent or adult requests: one teacher segment gently suggesting something
  else, with no bridge line.

Reply with one JSON object and nothing else:
{"segments": [{"instructions": "...", "utterance": "..."}]}

"instructions": 1-2 sentences on emotion, pace, pitch and energy, all consistent
(calm = slow and soft).
"utterance": plain spoken words only: no emojis, lists, quote marks around the
sentence, or emotion labels. Use straight quotes and escape any inside strings.

"""

try:
    whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
except Exception as e:
    print("model did not load", e)


_client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    #stream=True
    )

    #TODO: add try except here
client = AsyncHumeClient(api_key=os.environ["HUME_API_KEY"])
#TODO add key error try except here

VOICE = PostedUtteranceVoiceWithName(name="Ava Song", provider="HUME_AI")


def record(path="recording.wav"):
    """Record from the microphone until the speaker stops talking, and save it as a WAV file."""
    block_size = round(SAMPLE_RATE * BLOCK_SECONDS)   
    blocks = []               
    quiet_blocks = 0          
    started_talking = False

    print("Listening... speak now.")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as mic:
        while True:
            block, overflowed = mic.read(block_size)   
            blocks.append(block)
            loudness = np.abs(block).mean()            

            if loudness > SILENCE_LEVEL:
                started_talking = True
                quiet_blocks = 0
            else:
                quiet_blocks += 1

            if started_talking and quiet_blocks * BLOCK_SECONDS >= STOP_AFTER_SILENCE:
                break                                  
            if len(blocks) * BLOCK_SECONDS >= MAX_SECONDS:
                break                                  
    print("Recording finished.")

    audio = np.concatenate(blocks)                     
    sf.write(path, audio, SAMPLE_RATE)                 
    return path


def transcribe_local(path):
    """Turn a recording into text with faster-whisper."""
    audio, sample_rate = sf.read(path, dtype="float32")   
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
             "content": SYSTEM_PROMPT
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

{"segments": [{"instructions": "...", "utterance": "..."}]}

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
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

    clock = pygame.time.Clock()

    while pygame.mixer.music.get_busy():
        clock.tick(30)

    pygame.mixer.music.unload()

def split_sentences(text):
    """ Returns a list of strings """
    return re.split(r'(?<=[.!?])\s+', text)

async def speak_sentences(sentences: list, instructions: str, prefix: str = "response"):
    for i, sentence in enumerate(sentences, start=1):
        await speak(
            sentence,
            instructions,
            f'{prefix}{i:03d}.mp3')
        play_audio(f'{prefix}{i:03d}.mp3')


async def main():

    play_audio('introduction.mp3')

    memory1 = [] #adding memory

    while len(memory1) < 5: #limiting the memory
            if TEXT_MODE:
                    transcript = input("Type what you want to say: ")
            else:
                    record(path="recording.wav")
                    transcript = transcribe_local("recording.wav")

            print("Transcript:", transcript)


            response_llm, memory1 = get_LLM_response(transcript, memory1)
            print(response_llm)

            utterance1 = split_sentences(response_llm["segments"][0]["utterance"])
            await speak_sentences(utterance1, response_llm["segments"][0]["instructions"])

            if len(response_llm["segments"]) == 1:
                pass
            else:
                utterance2 = split_sentences(response_llm["segments"][1]["utterance"])
                await speak_sentences(utterance2, response_llm["segments"][1]["instructions"], prefix="response2")

    play_audio('conclusion.mp3')

    return None

if __name__ == "__main__":
    asyncio.run(main())
