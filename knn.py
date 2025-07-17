from dotenv import load_dotenv
from pathlib import Path
import os
import pandas as pd
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
dotenv_path = Path('.env.py')
load_dotenv(dotenv_path=dotenv_path)  # Load variables from .env file
print(__name__ == "__main__")
print(f"__name__ is: {__name__}")
app = Flask(__name__)
# it creates an instance of the Flask web application (__name__ holds the name of the current Python module.)
import numpy as np
import nltk
try:
    nltk.data.find("tokenizers/punkt")
    print("✅ punkt tokenizer already exists")
except LookupError:
    print("⬇ Downloading 'punkt' tokenizer...")
    nltk.download("punkt")
df = pd.read_csv('cleaned_data2.csv')
# print(df['Therapeutic Class'].value_counts())
# pain analgesic              2891
# anti infectives             1549
# respiratory                 1401
# gastro intestinal            926
# cardiac                      857
# neuro cns                    391
# vitamin mineral nutrient     250
# anti diabetic                221
# derma                        219
# anti malarials                81
# ophthal                       66
# gynaecological                45
# blood related                 32
# urology                       30
# otologicals                   20
# ophthal otologicals           10
# stomatologicals                4
# vaccine                        4
# hormone                        1
# anti neoplastics               1
# df.info()
# Data columns (total 4 columns):
# Column              Non-Null Count  Dtype
# ---  ------              --------------  -----
#  0   name                98557 non-null  object
#  1   short_composition1  98557 non-null  object
#  2   short_composition2  98557 non-null  object
#  3   Therapeutic Class   98557 non-null  object
# dtypes: object(4)
# memory usage: 3.0+ MB
names=df['name']
short_composition1=df['short_composition1']
short_composition2=df['short_composition2']
TherapeuticClass=df['Therapeutic Class']
# Tokenize all relevant columns (not every single column)
sentences = pd.concat([names, short_composition1, short_composition2]).astype(str).apply(nltk.word_tokenize).tolist()
# Train Word2Vec on full vocabulary
model = Word2Vec(sentences, vector_size=8999, min_count=1)
# Create a helper to get averaged vectors
def get_vector(text, model):
    words = nltk.word_tokenize(str(text))
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
# Transform input features
X_name = names.apply(lambda x: get_vector(x, model))
X_comp1 = short_composition1.apply(lambda x: get_vector(x, model))
X_comp2 = short_composition2.apply(lambda x: get_vector(x, model))
# Stack into a single feature matrix
X = np.stack((X_name + X_comp1 + X_comp2), axis=0).T
# Train the KNeighborsClassifier
Y = TherapeuticClass
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
knn = KNeighborsClassifier(n_neighbors=95)
knn.fit(X_train, Y_train)
print("Accuracy:", knn.score(X_test, Y_test))
# Xinput=["zogrell a 75mg75mg tablet","aspirin 75mg","clopidogrel 75mg"]
# Xmodel = Word2Vec(Xinput, vector_size=8999, min_count=1)
# X_input=[]
# for i in range (len(Xinput)):
#     X_input.append(get_vector(Xinput[i], Xmodel))
# print(knn.predict(X_input))
# Accuracy: 0.32555555555555554
# ['pain analgesic' 'pain analgesic' 'pain analgesic']
@app.route("/")
def health_check():
    return "OK", 200
@app.route('/predict', methods=['POST'])
def predict():
    input=[]
    name,composition1,composition2= request.json
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

    print("🚀 Starting Flask app...")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
# install waitress for run the app cuz flask is in developer mode
# waitress-serve --listen=127.0.0.1:5000 knn:app (run)
# pip freeze > requirements.txt will override the req file