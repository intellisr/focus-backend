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
# from flask_socketio import SocketIO
# from threading import Lock

from datetime import datetime

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

# sio = SocketIO(app)
# thread = None
# thread_lock = Lock()

cors_origins = []
CORS(app, resources={r"/*": {"origins": cors_origins}})

fileName="lecture12.txt"
    
@app.route("/proccess", methods=["GET"])
@cross_origin()
def proc():
    """proccess audio file"""

    ref = db.reference("/alldata/lecture1/")
    refStatus = db.reference("/status")
    refStatus.set(10)

    trans = Tmodel.transcribe('audio.wav')
    document= trans["text"]
    refStatus.set(20)

    # document="""As you have probably noticed, AI is currently a “hot topic”: media coverage and public discussion about AI is almost impossible to avoid. However, you may also have noticed that AI means different things to different people. For some, AI is about artificial life-forms that can surpass human intelligence, and for others, almost any data processing technology can be called AI.To set the scene, so to speak, we’ll discuss what AI is, how it can be defined, and what other fields or technologies are closely related. Before we do so, however, we’ll highlight three applications of AI that illustrate different aspects of AI. We’ll return to each of them throughout the course to deepen our understanding.
    # Self-driving cars require a combination of AI techniques of many kinds: search and planning to find the most convenient route from A to B, computer vision to identify obstacles, and decision making under uncertainty to cope with the complex and dynamic environment. Each of these must work with almost flawless precision in order to avoid accidents.
    # The same technologies are also used in other autonomous systems such as delivery robots, flying drones, and autonomous ships.
    # Implications: road safety should eventually improve as the reliability of the systems surpasses human level. The efficiency of logistics chains when moving goods should improve. Humans move into a supervisory role, keeping an eye on what’s going on while machines take care of the driving. Since transportation is such a crucial element in our daily life, it is likely that there are also some implications that we haven’t even thought about yet.
    # A lot of the information that we encounter in the course of a typical day is personalized. Examples include Facebook, Twitter, Instagram, and other social media content; online advertisements; music recommendations on Spotify; movie recommendations on Netflix, HBO, and other streaming services. Many online publishers such as newspapers’ and broadcasting companies’ websites as well as search engines such as Google also personalize the content they offer.
    # While the frontpage of the printed version of the New York Times or China Daily is the same for all readers, the frontpage of the online version is different for each user. The algorithms that determine the content that you see are based on AI.Implications: while many companies don’t want to reveal the details of their algorithms, being aware of the basic principles helps you understand the potential implications: these involve so called filter bubbles, echo-chambers, troll factories, fake news, and new forms of propaganda."""     

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

        print("success")
        refStatus.set(0)
    return jsonify({"status":True})

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