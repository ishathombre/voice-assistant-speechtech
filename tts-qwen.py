import torch, soundfile as sf, subprocess
from qwen_tts import Qwen3TTSModel

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map=device,
    dtype=torch.float16 if device == "mps" else torch.float32,
)

EMOTIONS = {
    "neutral":   "",
    "happy":     "Very happy.",
    "sad":       "Very sad.",
    "angry":     "Very angry.",
    "fearful":   "Scared and anxious, voice trembling.",
    "surprised": "Very surprised and astonished.",
    "disgusted": "Disgusted and repulsed.",
    
}
SENTENCES = [
    "She said she would be here by noon.",
    "The meeting has been moved to Thursday.",
    "I put the keys on the kitchen table.",
]
# for each sentence x emotion: generate, save as f"{model}_{emotion}_{i}.wav", time it
INSTRUCT = {
    "neutral": "A calm adult woman reading from a script. Even pace, steady "
               "medium volume, no emotional coloring, clear and professional.",

    "happy": "A young woman who has just heard great news and can't hide her "
             "smile. Bright, warm tone, slightly fast, pitch rising at the "
             "ends of phrases, a little laughter in the voice.",

    "sad": "A woman in her thirties speaking quietly after a loss. Slow, soft "
           "and low in volume, heavy pauses between phrases, voice thin and "
           "slightly trembling, trailing off at the end of the sentence.",

    "angry": "A woman who has been pushed too far and is holding back a "
             "shout. Tight, clipped delivery, sharp consonants, volume "
             "building through the line, breathing hard between phrases.",

    "fearful": "A woman whispering in a dark house, afraid someone will hear. "
               "Fast, shallow breaths, trembling voice, hushed volume, "
               "stopping mid-phrase to listen.",

    "surprised": "A woman who just walked into a surprise party. Voice jumps "
                 "up in pitch on the first word, a quick gasp, then a "
                 "breathless, incredulous delivery.",
}
print("here!")

for emotions in EMOTIONS:
    wavs, sr = model.generate_custom_voice(
    text="She said she would be here by noon.",
    language="English",
    speaker="Ryan",
    instruct=EMOTIONS[emotions],
)
    sf.write(f"{emotions}_qwen1.wav", wavs[0], sr)
    subprocess.run(["afplay", f"{emotions}_qwen1.wav"]) 


"""
wavs, sr = model.generate_custom_voice(text="She said she would be here by noon.",
    language="English",
    speaker="Ryan",
    instruct="A woman in her thirties speaks quietly after a loss, slow and soft with heavy pauses, her voice thin and trembling, trailing off at the end of the sentence.",
)
sf.write(f"happy_whisper.wav", wavs[0], sr)
subprocess.run(["afplay", "happy_whisper.wav"]) 

"""
  # plays through your Mac speakers
