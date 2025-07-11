import streamlit as st
import pandas as pd
from nlp_utils.preprocess import clean_text
from nlp_utils.predict import load_model, predict_sentiment, predict_sentiment_proba

# Load models
model = load_model("models/logistic_regression_model.pkl")
vectorizer = load_model("models/tfidf_vectorizer.pkl")

# UI setup
st.set_page_config(page_title="Amazon Review Sentiment App", layout="centered")
st.title("Amazon Review Sentiment Classifier")
st.markdown("Predict sentiment of product reviews (Positive, Neutral, Negative).")

# Emoji mapping
label_map = {0: "Negative 😠", 1: "Neutral 😐", 2: "Positive 😊"}

# --- Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info("""
This app uses a pre-trained model to classify the sentiment of Amazon product reviews.

- **Model**: TF-IDF + Logistic Regression
- **Data**: Amazon product reviews dataset
- **Source**: [GitHub Repository](#)
""")

# Display example metrics
st.sidebar.subheader("Model Metrics")
st.sidebar.metric("Accuracy", "82.5%")
st.sidebar.metric("F1 Score", "0.83")
st.sidebar.metric("Vectorizer", "TF-IDF")

# --- Single Review Prediction ---
st.subheader("🔍 Single Review Analysis")
review = st.text_area("✍️ Enter your product review here:", height=150)

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review)
        label, prob = predict_sentiment_proba(cleaned, vectorizer, model)
        st.success(f"Predicted Sentiment: **{label_map[label]}**")
        st.info(f"Confidence: {prob*100:.2f}%")

# --- Batch Prediction from CSV ---
st.subheader("📤 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'review' not in df.columns:
            st.error("CSV must contain a 'review' column.")
        else:
            df['cleaned'] = df['review'].apply(clean_text)
            results = df['cleaned'].apply(lambda x: predict_sentiment_proba(x, model, vectorizer))
            df['label'], df['confidence'] = zip(*results)
            df['label'] = df['label'].map(label_map)

            st.write("### Prediction Results")
            st.dataframe(df[['review', 'label', 'confidence']])

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="sentiment_predictions.csv",
                mime='text/csv'
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")
