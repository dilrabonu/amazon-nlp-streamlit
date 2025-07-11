# amazon-nlp-streamlit
 Streamlit app for Amazon product review sentiment classification using TF-IDF + Logistic Regression.

 # 🔍 Amazon Review Sentiment Classifier (Streamlit App)

This is an NLP-powered web app that predicts sentiment (Positive, Neutral, Negative) from Amazon product reviews using a Logistic Regression model trained on TF-IDF vectors.

## 💻 Features

- Cleaned using advanced NLP techniques (lemmatization, stopword removal)
- TF-IDF vectorizer + Logistic Regression
- Deployed via Streamlit for instant prediction
- Modular code for reuse and extension

## 📁 Project Structure

amazon-nlp-streamlit/
│
├── app.py
├── requirements.txt
├── README.md
├── /models/
│ ├── tfidf_model.pkl
│ └── tfidf_vectorizer.pkl
├── /nlp_utils/
│ ├── preprocess.py
│ └── predict.py


## 🚀 To Run

```bash
pip install -r requirements.txt
streamlit run app.py
📌 Author: Dilrabo Khidirova


---

## 🟢 Final Instructions

1. **Place your `.pkl` models** into `/models/`
2. **Run locally** in VS Code:
```bash
streamlit run app.py
