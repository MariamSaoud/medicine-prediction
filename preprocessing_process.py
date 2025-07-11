import pandas as pd     # load the data and perform cleaning data
import re       # regular expressions
import nltk     # Natural Language Toolkit (performing text data performing)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize     # limitization purpose
contractions = {
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}
import warnings # to ignore any types of warnings
warnings.filterwarnings('ignore')
# using the pandas to read the dataset csv file once it's loaded (will execute the df.head())
df = pd.read_csv("D:\\Extensive_A_Z_medicines_dataset_of_India.csv")

print(df.head())   # display the top 5 rows of the dataset (with all it's columns)  [5 rows x 24 columns]

# select specific columns because i don't want all of it (the columns that i didn't write it will ignore)
df_clean=df[['name','short_composition1', 'short_composition2','Therapeutic Class']]   # don't forget the inner square brackets
print(df_clean.head())    # [5 rows x 4 columns]
print(df_clean.shape)     # (256476, 4)

print(df_clean.info())     # check the information about the dataset
#   Column              Non-Null Count   Dtype
# ---  ------              --------------   -----
#  0   name                256476 non-null  object
#  1   short_composition1  256476 non-null  object
#  2   short_composition2  113099 non-null  object
#  3   Therapeutic Class   228317 non-null  object
# dtypes: object(4)
# memory usage: 7.8+ MB
# None
# checking for any missing values in our data set there isn't any missing values
print(df_clean.isna().sum())
# name                       0
# short_composition1         0
# short_composition2    143377
# Therapeutic Class      28159
# dtype: int64
# drop missing values
df_clean=df_clean.dropna()
print(df_clean.shape)     # (100655, 4)
# # checking if there is duplicating values
print(df_clean.duplicated().sum())        # 2098 number of duplicated rows
df_clean=df_clean.drop_duplicates()
print(df_clean.shape)     # (98557, 4)
def clean_text(text,remove_stop_word=True):
    text=text.lower()
    if True:
        text=text.split()
        new_text=[]
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text=" ".join(new_text)
        text
    # removing the URLS
    text=re.sub(r'https?:\/\/.*[\r\n]*','',text,flags=re.MULTILINE)
    # removing the username
    text=re.sub(r'@[A-Za-z0-9]+','',text)
    # removing HTML
    text=re.sub(r'\<a href','',text)
    text=re.sub(r'&amp;','',text)
    text=re.sub(r'[_"\;%()|+&=*%.,?!:#$@\[\]/]','',text)
    text=re.sub(r'<br/>','',text)
    text=re.sub(r'\'','',text)
    # tokenizing the text
    words=word_tokenize(text.lower())
    # remove stopwords if needed
    if remove_stop_word:
        stop_words=set(stopwords.words("english"))
    # limitization (back to the source words)
    limitizer=WordNetLemmatizer()
    words=[limitizer.lemmatize(word) for word in words]
    words=" ".join(words)
    return words
df_clean['name']=df_clean['name'].apply(lambda x:clean_text(x))
df_clean['short_composition1']=df_clean['short_composition1'].apply(lambda x:clean_text(x))
df_clean['short_composition2']=df_clean['short_composition2'].apply(lambda x:clean_text(x))
df_clean['Therapeutic Class']=df_clean['Therapeutic Class'].apply(lambda x:clean_text(x))
# print(df_clean.head())
# print the classes that contains in the dataset
# print(df['Therapeutic Class'].value_counts())
# Therapeutic Class
# anti infectives               18950
# pain analgesic                18857
# gastro intestinal             16283
# respiratory                   14668
# anti diabetic                  6862
# cardiac                        6504
# neuro cns                      6348
# derma                          3853
# vitamin mineral nutrient       2064
# ophthal                        1101
# blood related                   755
# gynaecological                  479
# ophthal otologicals             469
# urology                         399
# anti malarials                  393
# otologicals                     270
# stomatologicals                 145
# sex stimulant rejuvenators       64
# vaccine                          56
# anti neoplastics                 19
# others                           10
# hormone                           8
# Name: count, dtype: int64
# print normalize results for the classes
# print(df_clean['Therapeutic Class'].value_counts(normalize=True))
# Therapeutic Class
# anti infectives               0.192275
# pain analgesic                0.191331
# gastro intestinal             0.165214
# respiratory                   0.148828
# anti diabetic                 0.069625
# cardiac                       0.065992
# neuro cns                     0.064409
# derma                         0.039094
# vitamin mineral nutrient      0.020942
# ophthal                       0.011171
# blood related                 0.007661
# gynaecological                0.004860
# ophthal otologicals           0.004759
# urology                       0.004048
# anti malarials                0.003988
# otologicals                   0.002740
# stomatologicals               0.001471
# sex stimulant rejuvenators    0.000649
# vaccine                       0.000568
# anti neoplastics              0.000193
# others                        0.000101
# hormone                       0.000081
# Name: proportion, dtype: float64
df_clean['Therapeutic Class'].value_counts(normalize=True)
# df_clean.to_csv('cleaned_data.csv', index=False)