import streamlit as st
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer  

models = {
    "Naive Bayes": pickle.load(open("naive_bayes_model.pkl", "rb")),
    "SVM": pickle.load(open("svm_model.pkl", "rb")),
    "CNN": pickle.load(open("cnn_model.pkl", "rb")),    
    "BiLSTM": pickle.load(open("bilstm_model.pkl", "rb")),      
    "Random Forest": pickle.load(open("random_forest_model.pkl", "rb"))
}

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'[^a-z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100
st.title('Hate Speech Detection')
comment = st.text_area("Enter your comment:")

selected_model = st.selectbox(
    'Select a model',
    ['Naive Bayes', 'SVM', 'CNN', 'BiLSTM', 'Random Forest']
)

if st.button('Check'):
    comment = preprocess_text(comment)
    
    if selected_model == "Naive Bayes":
        vectorized_input = vectorizer.transform([comment])
    else:
        sequences = tokenizer.texts_to_sequences([comment])
        vectorized_input = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    model = models[selected_model]
    prediction = model.predict(vectorized_input)

    if prediction[0] > 0.5:
        result = "Hate"
    else:
        result = "Non-Hate"
    
    st.write(f"The comment is: {result}")
