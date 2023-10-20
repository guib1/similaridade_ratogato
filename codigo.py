from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

# Importe as stopwords em português do NLTK
nltk.download('stopwords')
from nltk.corpus import stopwords

# Defina as três strings de texto (Q, D1 e D2)
Q = "O Rato é Legal"
D1 = "O Gato é legal"
D2 = "Um Rato é caçado de forma legal"

Total = Q + " " + D1 + " " + D2

# Pré-processamento das strings
def preprocess(text):
    # Tokenização e remoção de stopwords
    words = [word for word in text.lower().split() if word not in stopwords.words('portuguese')]
    return ' '.join(words)

# Aplicar pré-processamento e criar um vetorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([preprocess(Q), preprocess(D1), preprocess(D2)])

# Calcule a similaridade de cossenos entre Q e D1
similarity_Q_D1 = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

# Calcule a similaridade de cossenos entre Q e D2
similarity_Q_D2 = cosine_similarity(tfidf_matrix[0], tfidf_matrix[2])

print("Similaridade de cossenos entre Q e D1:", similarity_Q_D1[0][0])
print("Similaridade de cossenos entre Q e D2:", similarity_Q_D2[0][0])

palavras = [word for word in Total.lower().split() if word not in stopwords.words('portuguese')]
palavras_sem_repeticao = list(set(palavras))

print(palavras_sem_repeticao)