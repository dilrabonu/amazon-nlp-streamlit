# 📁 predict.py

import joblib
import numpy as np

def load_model(path):
    """
    Load a pre-trained model from the specified path.
    """
    return joblib.load(path)

def predict_sentiment(model, vectorizer, text):
    """
    Predict the sentiment class (0, 1, 2).
    """
    X = vectorizer.transform([text])
    return model.predict(X)[0]  # Return class label

def predict_sentiment_proba(text, vectorizer, model):
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    label = int(proba.argmax())
    return label, proba[label]
