import asyncio
import re
import time

from asr import transcribe_local
from audio import play_audio, record
from llm import get_LLM_response
from clients import TEXT_MODE
from tts import speak_sentences

def split_sentences(text: str):
    """ Returns a list of strings """
    return re.split(r'(?<=[.!?])\s+', text)

async def main():

    #timing the overall pipeline
    t_pipeline = time.perf_counter()

    try:
        play_audio('introduction.mp3')
    except Exception as e:
        print(e, "Continuing without playing the introduction")

    memory1 = []

    while len(memory1) < 5: #limiting the memory
            asr_secs = 0.0

            if TEXT_MODE:
                    transcript = input("Type what you want to say: ")
            else:
                    record(path="recording.wav")
                    #adding time block
                    t_asr = time.perf_counter()
                    transcript = transcribe_local("recording.wav")
                    asr_secs = time.perf_counter() - t_asr

            print("Transcript:", transcript)


            #timing the llm section of the pipeline
            t_llm = time.perf_counter()
            response_llm, memory1 = get_LLM_response(transcript, memory1)
            llm_secs = time.perf_counter() - t_llm

            #timing the overall tts section

            t_tts = time.perf_counter()

            utterance1 = split_sentences(response_llm["segments"][0]["utterance"])
            await speak_sentences(utterance1, response_llm["segments"][0]["instructions"])

            if len(response_llm["segments"]) == 1:
                pass
            else:
                utterance2 = split_sentences(response_llm["segments"][1]["utterance"])
                await speak_sentences(utterance2, response_llm["segments"][1]["instructions"], prefix="response2_")

            tts_secs = time.perf_counter() - t_tts


            print("===Time taken for current converations===")
            print(f"Time taken for ASR: {asr_secs}")
            print(f"Time taken for TTS: {tts_secs}")
            print(f"Time taken for LLM: {llm_secs}")
            print(f"Overall time taken: {time.perf_counter() - t_pipeline}")

    try:
        play_audio('conclusion.mp3')
    except Exception as e:
        print(e, "Ending the conversation!")

    return None

if __name__ == "__main__":
    asyncio.run(main())
