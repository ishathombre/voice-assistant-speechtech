import os
import time
#import playsound
import speech_recognition as sr
from gtts import gTTS
import subprocess
import simpleaudio, time
from groq import Groq


    

#strong_beat = simpleaudio.WaveObject.from_wave_file('strong_beat.wav')
#weak_beat = simpleaudio.WaveObject.from_wave_file('weak_beat.wav')




def speak(text):
    tts = gTTS(text = text, lang = "en")
    filename = "voice.mp3"
    tts.save(filename)
    subprocess.call(["afplay", filename])

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""
        try:
            said = r.recognize_google(audio)
            print(said)
        except Exception as e:
            print(f"execption, {e}")
    return said


def ask_groq(text):
    client = api_key= os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": text,
        }
      ],
      model="openai/gpt-oss-120b",)
    print(chat_completion.choices[0].message.content)
    speak(chat_completion.choices[0].message.content)
    


#TODO: integrate wake one over here

speak("hi i am your voice assistant!")
text = get_audio()
ask_groq(text)
flag = 0
while flag == 0:
    text = get_audio()
    ask_groq(text)
    if "stop this now" in text:
        flag = 1
