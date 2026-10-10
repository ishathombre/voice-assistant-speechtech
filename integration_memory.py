import numpy as np
import asyncio, base64, time
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
#from faster_whisper import WhisperModel
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
TEXT_MODE = True     
PLAY_SOUND = True  
# Initialize external tools
load_dotenv()

SYSTEM_PROMPT = """
You are the brain of a friendly voice assistant for children. Everything you write is
spoken aloud by a text-to-speech (TTS) system that can act out emotions, so write the
way a kind teacher would talk.

WHAT THE USER MIGHT SEND
A) SAY REQUEST: they want a sentence said in a certain emotion.
B) QUESTION: they ask something, for example about an emotion, about how a sentence
   sounded, or about the conversation so far.
C) BOTH in one message.

HOW TO REPLY
Your reply is a list of segments, spoken in order.
- SAY REQUEST only: two segments.
  1. A short lead-in: "Hmmm, let's fix it."
  2. The sentence the user gave, in the emotion they asked for. Say it exactly as
     given, without the quotation marks and without the words naming the emotion.
- QUESTION only: two segments.
  1. The answer: at most 3 short, simple sentences. Use the earlier conversation when
     it is relevant.
  2. The sentence from the user's most recent say request, said again in the same
     emotion as before, unless the user asked for a different one.
  If there was no earlier say request, give only the answer segment.
- BOTH: one segment per part, in the order the user asked, with nothing extra.
- UNCLEAR, EMPTY OR NONSENSE: one segment with a short, friendly question asking the
  user to say it again or explain.
- UNSUITABLE (scary, violent, adult): one segment that gently suggests something else.

OUTPUT FORMAT
Reply with one JSON object and nothing else, in exactly this format:
{"segments": [{"instructions": "<how the voice should sound>", "utterance": "<the words to speak>"}]}

- "instructions": 1 to 2 sentences describing emotion, pace, pitch and energy. Make the
  parts agree with each other (for example, calm means slow and soft, not fast).
  For answers and lead-ins, use a warm, friendly teacher voice.
- "utterance": plain spoken words only. No emojis, lists, stage directions or emotion
  labels.
- Use straight quotes, and escape any quote marks inside a string.
- Keep everything child-friendly.

EXAMPLE 1 (say request)
User: Can you say the following sentence in a cheerful voice: "we ate this assignment."
{"segments": [
  {"instructions": "Warm, friendly teacher voice, gentle pacing.", "utterance": "Hmmm, let's fix it."},
  {"instructions": "Bright and cheerful, warm smile in the voice, light and bouncy pacing, pitch lifting on the important words.", "utterance": "We ate this assignment!"}
]}

EXAMPLE 2 (question after a say request)
Earlier, the user asked for "it's okay to try sometimes" in a calm voice, and you said it.
User: Why did that sound so slow?
{"segments": [
  {"instructions": "Warm, friendly teacher voice, gentle pacing.", "utterance": "Calm voices are slow and soft because that helps our bodies relax. Slow words feel safe and peaceful."},
  {"instructions": "Calm and soothing, slow pacing, soft pitch.", "utterance": "It's okay to try sometimes."}
]}

EXAMPLE 3 (both)
User: Why is a smile warm? Then say "you can do it" in an excited voice.
{"segments": [
  {"instructions": "Warm, friendly teacher voice, gentle pacing.", "utterance": "A smile feels warm because it shows kindness and makes other people happy too."},
  {"instructions": "Excited and energetic, fast pacing, rising pitch.", "utterance": "You can do it!"}
]}


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
            await speak("Getting into character and saying the sentence.", "Warm, friendly teacher voice, gentle pacing.", audio_response_path2)
               
            await speak(
                    response_llm["segments"][1]["utterance"],
                    response_llm["segments"][1]["instructions"],
                    audio_response_path1
                )

            
            play_audio(audio_response_path)
            play_audio(audio_response_path2)
            play_audio(audio_response_path1)
    await speak("That is it for now!", "Warm, friendly teacher voice, gentle pacing.", audio_response_path2)
    play_audio(audio_response_path2)

    return audio_response_path

if __name__ == "__main__":
    asyncio.run(main())
