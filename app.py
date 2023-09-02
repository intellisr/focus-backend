from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from pprint import pprint
# nltk.download('punkt')
from qna import generateMCQ
import whisper

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import threading

from datetime import datetime

import pyaudio
import wave
import time
from datetime import datetime
import whisper
import os

Tmodel = whisper.load_model("base.en")

model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

tokenizerTitle = AutoTokenizer.from_pretrained("fabiochiu/t5-small-medium-title-generation")
modelTitle = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-small-medium-title-generation")

# import firebase_admin
from firebase_admin import db,credentials,initialize_app

databaseURL="https://focus-77577-default-rtdb.firebaseio.com"
cred_obj = credentials.Certificate('focus-77577-firebase-adminsdk-bptde-a410d97c81.json')
default_app = initialize_app(cred_obj, {
	'databaseURL':databaseURL
	})


app = Flask(__name__)

# Global flag to control function execution
running_flag = False


cors_origins = []
CORS(app, resources={r"/*": {"origins": cors_origins}})

fileName="lecture1.txt"
lecture_action=False
clip_time=30

def voice_to_wav():
    global running_flag
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
    while running_flag:
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
    global running_flag
    while running_flag:
        file_names = os.listdir("audio")
        if len(file_names) > 1:
            full_path = ["audio/{0}".format(x) for x in file_names]  
            oldest_file = min(full_path, key=os.path.getctime)    
            subPro(oldest_file)
        if lecture_action:
            break

def subPro(name):
    result = Tmodel.transcribe(name)
    time_str=name.replace("audio-","").replace(".wav","")
    txt=result["text"]
    with open(fileName, 'a+') as f:
        f.write(time_str+"|"+txt)
        f.write('\n')
        f.close()
    os.remove(name)

function_thread = threading.Thread(target=voice_to_wav)
function_thread2 = threading.Thread(target=wav_to_txt)

@app.route('/start', methods=['GET'])
@cross_origin()
def start_function():
    global running_flag, function_thread,function_thread2
    if not running_flag:
        os.remove('audio-main.wav')
        file_names = os.listdir("audio")
        for name in file_names:
            os.remove(name)
        f = open(fileName, 'r+')
        f.truncate(0)
        running_flag = True
        function_thread.start()
        function_thread2.start()
        return jsonify(message="Function started.")
    else:
        return jsonify(message="Function is already running.")

@app.route('/stop', methods=['GET'])
@cross_origin()
def stop_function():
    global running_flag, function_thread,function_thread2
    if running_flag:
        running_flag = False
        function_thread.join()  # Wait for the function thread to finish
        return jsonify(message="Function stopped.")
    else:
        return jsonify(message="Function is not running.")
    
@app.route("/proccess", methods=["GET"])
@cross_origin()
def proc():
    """proccess audio file"""

    with open(fileName, 'rt') as file:
        lines=file.readlines()
        refStatus.set(10)
    if len(lines) > 2:    

        ref = db.reference("/alldata/lecture1/")
        refStatus = db.reference("/status")

        trans = Tmodel.transcribe('audio-main.wav')
        document= trans["text"]
        refStatus.set(20)

        # separate the text into sentences
        sentences = tokenizer.tokenize(document)

        # create initial embeddings for comparison
        prev_sentence_embedding = model.encode([sentences[0]])[0]
        passages = []
        passage = sentences[0]

        for sentence in sentences[1:]:
            current_sentence_embedding = model.encode([sentence])[0]
            if cosine_similarity([prev_sentence_embedding], [current_sentence_embedding]) < 0.2:
                # append the passage to passages and create a new passage
                passages.append(passage)
                passage = sentence
            else:
                # continue adding sentences to the current passage
                passage = passage + " " + sentence
            # update previous sentence embedding
            prev_sentence_embedding = current_sentence_embedding

        # Make sure to add the final passage to the list
        passages.append(passage)
        refStatus.set(40)

        max_input_length=2000

        passageWithTitles={}
        num=0
        for x in passages:
            num=num+1

            inputs = ["summarize: " + x]

            inputs = tokenizerTitle(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
            output = modelTitle.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=30)
            decoded_output = tokenizerTitle.batch_decode(output, skip_special_tokens=True)[0]
            predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

            passageWithTitles.update({predicted_title:x})
            refStatus.set(60)
            data=generateMCQ(x)
            refStatus.set(80)
            ref = db.reference("/alldata/lecture1/passage-"+str(num)+"/")
            ref.set({
                "Name":predicted_title,
                "Passage":x,
                "QNA":data
            })
            refStatus.set(100)

            print("SUCCESS")
            refStatus.set(0)

            st=True
    else:
            refStatus.set(100)
            st=False
            print("FAILED")
            refStatus.set(0)

    return jsonify({"status":st})

@app.route("/dashboard", methods=["GET"])
@cross_origin()
def dash():
    """get data for dash board"""

    refStatus = db.reference("/dashstatus")
    refStatus.set(10)

    ref = db.reference('/alldata/lecture1')
    passages=ref.get()
    ref2 = db.reference('/usersLive/lecture1')
    attention=ref2.get()

    passages_for_text={}
    with open(fileName, 'rt') as file:  # Open the text file in read-text mode
        for line in file.readlines():
            textline=line.split("|")[1]
            time=line.split("|")[0]
            textlineEmbedding=model.encode([textline.strip()])[0]
            passagewiselist=[]            
            for passage in passages.values():
                full_passage=passage['Passage']
                full_passageEmbedding=model.encode([full_passage])[0]
                cosine_sim=cosine_similarity([full_passageEmbedding], [textlineEmbedding])
                passagewiselist.append(cosine_sim)

            pindex=passagewiselist.index(max(passagewiselist))
            passages_for_text[time]=pindex+1

    date_format = '%Y-%m-%d %H:%M:%S.%f'
    refStatus.set(30)

    grouped_dict = {}
    for key, value in passages_for_text.items():
        if value not in grouped_dict:
            ak=key.split("/")[1]
            date_obj = datetime.strptime(ak, date_format)
            grouped_dict[value] = [date_obj.replace(microsecond=0)]
        else:
            ak=key.split("/")[1]
            date_obj = datetime.strptime(ak, date_format)
            grouped_dict[value].append(date_obj.replace(microsecond=0))

    refStatus.set(60)        

    passageViseDict={}
    for key, value in grouped_dict.items():
        print("passage:"+str(key))
        youngust = min(value)
        oldest= max(value)
        print(youngust,oldest)
        stViseDict={}
        for keyv, value in attention.items():
            stList=[]
            for key2, value2 in value.items():
                dt = datetime.strptime(value2['time'], "%a, %d %b %Y %H:%M:%S %Z")
                if youngust < dt < oldest:
                    #print(youngust,dt,oldest)
                    stList.append(value2['data'])
            if len(stList) > 0:
                presntage=(sum(stList)/len(stList))*100      
                stViseDict[keyv]=presntage
            else:
                stViseDict[keyv]=0
            
        passageViseDict[key]=stViseDict

    refStatus.set(100)    
    refStatus.set(0) 
    return jsonify(passageViseDict)          
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=8000)     