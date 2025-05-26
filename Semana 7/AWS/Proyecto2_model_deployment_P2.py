#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess_text_1(text):
    if isinstance(text, str):  # Asegurarse de que la entrada sea una cadena
        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
        try:
            WordNetLemmatizer().lemmatize('test')
        except LookupError:
            nltk.download('wordnet')
        try:
            nltk.word_tokenize("example")
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        text = text.lower()
        # Liempieza de caracteres diferentes a letras, sustitucion por " "
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(
            word) > 2]  # Lematización y eliminación de stopwords y palabras cortas
        return ' '.join(tokens)
    return ''  # Manejar casos donde el texto no es una cadena


def predict_genre(PLOT):

    model, vect = joblib.load(os.path.dirname(__file__) +
                              '/genres_movie.pkl')

    X = preprocess_text_1(PLOT)
    print('Preprocessed text:', X)
    X_vec = vect.transform([X])
    print(X_vec)

    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
            'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
            'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

    # Make prediction
    prediction = model.predict_proba(X_vec)

    return pd.DataFrame(prediction, columns=cols)
    # return X


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print('Please add an description of the movie as an argument.')

    else:

        plot = sys.argv[1]

        print('\nDescripción de entrada:')
        print(plot)

        genres_df = predict_genre(plot)

        # Ordenar por probabilidad descendente
        sorted_probs = genres_df.T.sort_values(by=0, ascending=False)
        sorted_probs.columns = ['Probabilidad']

        # Filtrar géneros con probabilidad mayor a un umbral (opcional)
        umbral = 0.1
        sorted_probs = sorted_probs[sorted_probs['Probabilidad'] > umbral]

        print('\n🎬 Géneros probables:')
        print(sorted_probs.to_string(float_format="%.3f"))
