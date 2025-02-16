import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import contractions
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim

import ssl

# Fix SSL issue for nltk downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

print("All necessary NLTK resources downloaded successfully.")

# Load Data
df = pd.read_csv('final_data_topic_modelling.csv', encoding='latin1')

# Data Cleaning
df = df.dropna(subset=['Content']).drop_duplicates(subset=['Content'])

def expand_contractions(text):
    return ' '.join([contractions.fix(word) for word in str(text).split()]).lower()
df['Content'] = df['Content'].apply(expand_contractions)

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', str(text))
df['Content'] = df['Content'].apply(remove_punctuation)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(str(text))
    filtered = [word for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join([lemmatizer.lemmatize(word) for word in filtered])
df['Content'] = df['Content'].apply(preprocess_text)

def count_words(text):
    return len(word_tokenize(text))
df['WordCount'] = df['Content'].apply(count_words)

# Tokenization for LDA
df['text_tokenized'] = df['Content'].apply(word_tokenize)
id2word = corpora.Dictionary(df['text_tokenized'])
corpus = [id2word.doc2bow(text) for text in df['text_tokenized']]

# LDA Model Training
lda_model = gensim.models.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=20,
    random_state=100,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha='auto',
    per_word_topics=True
)

# LDA Visualization for VSCode
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'lda_visualization.html')

print("LDA visualization saved as 'lda_visualization.html'. Open it in your browser.")


