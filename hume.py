from hume import AsyncHumeClient
from hume.tts import PostedUtterance, PostedUtteranceVoiceWithName

load_dotenv()
client = AsyncHumeClient(api_key=os.environ["HUME_API_KEY"])

# Same voice for every emotion, so only the delivery changes.
# "Ava Song" is the voice used in Hume's docs example.
VOICE = PostedUtteranceVoiceWithName(name="Ava Song", provider="HUME_AI")

# With a voice set, the description acts as "acting instructions"
EMOTIONS = {
    "neutral":   "Calm and even, no emotional coloring",
    "happy":     "Bright and cheerful, smiling as she speaks",
    "sad":       "Quiet and heavy, voice thin and trembling, trailing off",
    "angry":     "Tight and clipped, barely holding back a shout",
    "fearful":   "Whispered and breathless, afraid of being overheard",
    "surprised": "Sudden and breathless, pitch jumping up in disbelief",
}
SENTENCES = [
    "She said she would be here by noon.",
    "The meeting has been moved to Thursday.",
]

INSTRUCT = {
    "neutral": "A calm adult woman whispering a line from a script. Even pace, "
               "soft hushed breath, no emotional coloring, clear and steady",

    "happy": "A young woman whispering, barely containing a smile after great "
             "news. Light, warm hushed tone, slightly fast, pitch lifting at "
             "the ends of phrases, a breathy little laugh in the whisper.",

    "sad": "A woman in her thirties whispering after a loss. Slow and very "
           "soft, heavy pauses between phrases, breathy and thin with a "
           "slight tremble, the whisper trailing off at the end.",

    "angry": "A woman whispering through clenched teeth, holding back a shout. "
             "Tight, clipped hiss, sharp consonants, intensity building "
             "through the line, breathing hard between phrases.",

    "fearful": "A woman whispering in a dark house, afraid someone will hear. "
               "Fast, shallow breaths, trembling hushed voice, stopping "
               "mid-phrase to listen.",

    "surprised": "A woman whispering at a surprise party, caught off guard. "
                 "The first word jumps up in pitch, a quick gasp, then a "
                 "breathless, incredulous hushed delivery.",
}
async def speak(text, emotion, path):
    t0 = time.time()
    result = await client.tts.synthesize_json(
        utterances=[PostedUtterance(text=text, description=INSTRUCT[emotion], voice=VOICE)],
    )
    secs = time.time() - t0
    with open(path, "wb") as f:
        f.write(base64.b64decode(result.generations[0].audio))
    return secs

async def main():
    await speak("Warm up.", "neutral", "hume_warmup.mp3")   # not timed
    for i, text in enumerate(SENTENCES):
        for emo in EMOTIONS:
            path = f"hume_{emo}_s{i}_instruct_whispered.mp3"
            secs = await speak(text, emo, path)
            print(f"{emo:10s} s{i}  {secs:.1f}s -> {path}")
            await asyncio.sleep(4)   # stay under the free tier's request limit

asyncio.run(main())
