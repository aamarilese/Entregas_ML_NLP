#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os


def predict_popularity(X_test):

    stacking = joblib.load(os.path.dirname(__file__) +
                           '/popularity_stacking.pkl')

    X = pd.DataFrame([X_test])

    drop_cols = ['track_id', 'track_name',
                 'artists', 'album_name', 'track_genre']

    X = X.drop(columns=drop_cols)

    # Make prediction
    prediction = stacking.predict(X)

    return prediction


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print('Please add an URL')

    else:

        url = sys.argv[1]

        popularity_score = predict_popularity(url)

        print(url)
        print('Popularidad: ', popularity_score)
