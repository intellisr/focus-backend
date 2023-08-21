# from fastapi import FastAPI
import pyaudio
import wave
import time
from datetime import datetime
import whisper
import os
from threading import Thread

lecture_action=False
clip_time=30

model = whisper.load_model("base.en")

def voice_to_wav():
    # Create a PyAudio object
    p = pyaudio.PyAudio()

    # Open a stream to the audio input device
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    # Create an MP3 file object to save the audio
    now = datetime.now()
    wf = wave.open("audio/audio-"+str(now)+".wav", "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)

    wf2 = wave.open("audio-main.wav", "wb")
    wf2.setnchannels(1)
    wf2.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf2.setframerate(44100)
    

    # Start recording audio
    start_time=time.time()
    while True:
        # Get the audio data from the stream
        data = stream.read(1024)

        # Write the audio data to the wave file
        wf.writeframes(data)
        wf2.writeframes(data)

        # Check if 1 minutes have passed
        time_elapsed = time.time() - start_time
        if time_elapsed >= clip_time:
            # Start recording audio
            start_time = time.time()
            now = datetime.now()

            # Create a new wave file object
            wf = wave.open("audio/audio-"+str(now)+".wav", "wb")
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)

            if lecture_action:
                # Stop recording audio
                stream.stop_stream()
                stream.close()
                break

def wav_to_txt():
    while True:
        file_names = os.listdir("audio")
        if len(file_names) > 1:
            full_path = ["audio/{0}".format(x) for x in file_names]  
            oldest_file = min(full_path, key=os.path.getctime)    
            subPro(oldest_file)
        if lecture_action:
            break

def subPro(name):
    result = model.transcribe(name)
    time_str=name.replace("audio-","").replace(".wav","")
    txt=result["text"]
    with open('lecture1.txt', 'a+') as f:
        f.write(time_str+"|"+txt)
        f.write('\n')
        f.close()
    os.remove(name)

if __name__ == '__main__':
    Thread(target = voice_to_wav).start()
    Thread(target = wav_to_txt).start()
