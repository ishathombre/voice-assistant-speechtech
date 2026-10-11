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

# Dev switches
TEXT_MODE = False     
PLAY_SOUND = True  
# Initialize external tools
load_dotenv()

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

    return tts_input,memory




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
    audio_response_path1 = "sample_audio1.mp3"
    audio_response_path2 = "sample_audio2.mp3"
    
    #TODO: to reduce latency, save it in a sample_audio_introduction.mp3, so we don't need to call it
    await speak("Hi! I am your Emotion Tutor! To teach you about the world of emotions! We are going to learn about an emotion and create a short story together! Are you ready? Put your director hat, and I will act out your instructions!", "Warm, friendly teacher voice, gentle pacing.", audio_response_path2)
    play_audio(audio_response_path2)
    
    memory1 = [] #adding memory

    while len(memory1) < 5: #limiting the memory
            if TEXT_MODE:
                    transcript = input("Type what you want to say: ")
            else:
                    record(path="recording.wav")
                    transcript = transcribe_local("recording.wav")
            
            print("Transcript:", transcript)
            
            
            response_llm,memory1 = get_LLM_response(transcript, memory1)
            print(response_llm)
        
        
            await speak(
        response_llm["segments"][0]["utterance"],
        response_llm["segments"][0]["instructions"],
        audio_response_path
    )
            play_audio(audio_response_path)
    
            if len(response_llm["segments"]) == 1:
                pass
            else:
               
                await speak(
                    response_llm["segments"][1]["utterance"],
                    response_llm["segments"][1]["instructions"],
                    audio_response_path1
                )
                play_audio(audio_response_path1)
            

            
            
        
    #TODO: to reduce latency, save it in a sample_audio_conclusion.mp3, so we don't need to call it    
    await speak("That is it for now!", "Warm, friendly teacher voice, gentle pacing.", audio_response_path2)
    play_audio(audio_response_path2)

    return audio_response_path

if __name__ == "__main__":
    asyncio.run(main())
