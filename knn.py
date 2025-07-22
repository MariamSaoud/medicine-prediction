from dotenv import load_dotenv
from pathlib import Path
import os
import pandas as pd
from flask import Flask, request, jsonify
dotenv_path = Path('.env.py')
load_dotenv(dotenv_path=dotenv_path)  # Load variables from .env file
print("Current working directory:", os.getcwd())
print(f"__name__ is: {__name__}")
app = Flask(__name__)
# it creates an instance of the Flask web application (__name__ holds the name of the current Python module.)
import numpy as np
import nltk
nltk.data.path.append('./nltk_data')
try:
    nltk.data.find("tokenizers/punkt_tab")
    print("✅ punkt_tab tokenizer already exists")
except LookupError:
    print("⬇ Downloading 'punkt_tab' tokenizer...")
    nltk.download("punkt_tab")

try:
    nltk.data.find("tokenizers/punkt")
    print("✅ punkt tokenizer already exists")
except LookupError:
    print("⬇ Downloading 'punkt' tokenizer...")
    nltk.download("punkt")
df = pd.read_csv('cleaned_data2.csv')
from gensim.models import Word2Vec
import joblib

print("knn_model.pkl exists:", os.path.exists("knn_model.pkl"))
print("File size:", os.path.getsize("knn_model.pkl"))
# model = Word2Vec.load("word2vec.model")
knn = joblib.load("knn_model.pkl")
def get_vector(text, model):
    words = nltk.word_tokenize(str(text))
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
@app.route("/")
def health_check():
    return jsonify({'result':'OK'})
@app.route('/predict', methods=['POST'])
def predict():
    input=[]
    data = request.get_json()
    name = data.get("name")
    composition1 = data.get("composition1")
    composition2 = data.get("composition2")
    input.append(name)
    input.append(composition1)
    input.append(composition2)
    inputModel = Word2Vec(input, vector_size=8999, min_count=1)
    X_input = []
    for i in range(len(input)):
        X_input.append(get_vector(input[i], inputModel))
    c=knn.predict(X_input)
    return jsonify({'result':c[0]})
if __name__ == "__main__":
    try:
        nltk.data.find("tokenizers/punkt")
        print("✅ punkt tokenizer already exists")
    except LookupError:
        print("⬇ Downloading 'punkt' tokenizer...")
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
        print("✅ punkt_tab tokenizer already exists")
    except LookupError:
        print("⬇ Downloading 'punkt_tab' tokenizer...")
        nltk.download("punkt_tab")
    print("🚀 Starting Flask app...")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
# install waitress for run the app cuz flask is in developer mode
# waitress-serve --listen=127.0.0.1:5000 knn:app (run)
# pip freeze > requirements.txt will override the req file