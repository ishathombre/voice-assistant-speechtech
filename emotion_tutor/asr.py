import soundfile as sf
from clients import whisper_model

def transcribe_local(path):
    """Turn a recording into text with faster-whisper."""
    audio, sample_rate = sf.read(path, dtype="float32") 
    segments, info = whisper_model.transcribe(audio)
    return " ".join(segment.text.strip() for segment in segments)
