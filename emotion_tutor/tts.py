import base64
import time

from audio import play_audio
from clients import VOICE, hume_client
from hume.tts import PostedUtterance

async def speak(text, instruct, path):
    t0 = time.time()
    result = await hume_client.tts.synthesize_json(
        utterances=[PostedUtterance(text=text, description=instruct, voice=VOICE)],
    )
    secs = time.time() - t0
    with open(path, "wb") as f:
        f.write(base64.b64decode(result.generations[0].audio))
    return secs


async def speak_sentences(sentences: list, instructions: str, prefix: str = "response"):
    for i, sentence in enumerate(sentences, start=1):
        await speak(
            sentence,
            instructions,
            f'{prefix}{i:03d}.mp3')
        play_audio(f'{prefix}{i:03d}.mp3')
