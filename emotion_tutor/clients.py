import os

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from hume import AsyncHumeClient
from hume.tts import PostedUtteranceVoiceWithName
from openai import OpenAI

load_dotenv()

# Default dev switch state
TEXT_MODE = False
PLAY_SOUND = True

# 1. Whisper
try:
    whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
except Exception as e:
    print("model did not load, switching into text mode....", e)
    TEXT_MODE=True

# 2. DeepSeek
try:
    deepseek_client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )
except Exception as e:
    raise SystemExit(f"{e}..Could not load DEEPSEEK API key, exiting....")


# 3. Hume
try:
    hume_client = AsyncHumeClient(api_key=os.environ["HUME_API_KEY"])
except Exception as e:
    raise SystemExit(f"{e}..Could not load HUME API key, exiting....")

VOICE = PostedUtteranceVoiceWithName(name="Ava Song", provider="HUME_AI")
