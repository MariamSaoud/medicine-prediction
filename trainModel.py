import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
import nltk
import os
print("✅ Models saved to:", os.getcwd())
import traceback

print("🚦 Starting model training...")

try:
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
    df = pd.read_csv("cleaned_data2.csv")
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
    names = df["name"].astype(str)
    comp1 = df["short_composition1"].astype(str)
    comp2 = df["short_composition2"].astype(str)
    TherapeuticClass = df["Therapeutic Class"]
    # Tokenize all relevant columns (not every single column)
    # Train Word2Vec on full vocabulary
    sentences = pd.concat([names, comp1, comp2]).apply(nltk.word_tokenize).tolist()
    model = Word2Vec(sentences, vector_size=8999, min_count=1)
    model.save("word2vec.model")
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
    X_comp1 = comp1.apply(lambda x: get_vector(x, model))
    X_comp2 = comp2.apply(lambda x: get_vector(x, model))
    # Stack into a single feature matrix
    X = np.stack((X_name + X_comp1 + X_comp2), axis=0).T
    Y = TherapeuticClass
    # Train the KNeighborsClassifier
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
    # Its important to use binary mode
    knnPickle = open('knnpickle_file.pkl', 'wb')
    joblib.dump(knn, knnPickle)
    # close the file
    knnPickle.close()
    print("✅ Finished training")
except Exception as e:
    print("❌ Error during training:", e)
    traceback.print_exc()