# pip install --upgrade --verbose git+https://github.com/ramsrigouthamg/Questgen.ai.git
# pip install --quiet git+https://github.com/boudinfl/pke.git
# python -m nltk.downloader universal_tagset
# python -m spacy download en_core_web_sm
# wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
# tar -xvf  s2v_reddit_2015_md.tar.gz
# ls s2v_old


import nltk
# nltk.download('stopwords')
from Questgen import main

def generateMCQ(text):
    payload = {
                "input_text": text
            }

    qg = main.QGen()
    output = qg.predict_mcq(payload)

    return output