import flask
import pandas as pd
import pandas as pd
import numpy as np
import numpy as np
import codecs
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
import re
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
import nltk
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model 
from flask import Flask, request, render_template
import streamlit as st



model = load_model("mlfinalmodel.h5") 
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

label2int = {'0': 0, '1': 1}
int2label = {0: '0', 1: '1'}
SEQUENCE_LENGTH = 100 

def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', str(sentence))
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', str(sentence))
    sentence = re.sub(r'\s+', ' ', str(sentence))

    return sentence

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', str(text))

def get_predictions(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    # get the prediction
    prediction = model.predict(sequence)[0]
    # one-hot encoded vector, revert using np.argmax
    return int2label[np.argmax(prediction)]

from PIL import Image


st.title("Check whether you are depressed or not using your social media posts")
img=Image.open('bias-in-machine-learning.jpg')
st.image(img,width=700,output_format="auto",hight=100)
	# st.subheader("ML App with Streamlit")
html_temp = """
	<div style="background-color:gray;padding:10px">
	<h1 style="color:white;text-align:center;">Depression Detection App </h1>
	</div>
	"""
st.markdown(html_temp,unsafe_allow_html=True)

# Get user input
user_input =st.text_input("TYPE YOUR POST BELOW","Enter your post")
user_input=[user_input]
user_input=tokenizer.texts_to_sequences(user_input)
user_sequence = pad_sequences(user_input)
user_prediction =model.predict(user_sequence)

if st.button("Predict"):
    if np.around(user_prediction, decimals=0)[0][0] == 1.0:
        st.write('You are depressed.Please visit the counselor')
    else:
        st.write("You are not depressed")