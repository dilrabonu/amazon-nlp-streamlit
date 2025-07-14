import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import WhitespaceTokenizer
tokenizer = WhitespaceTokenizer()


nltk.data.path.append("C:/Users/user/AppData/Roaming/nltk_data")


#  Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Preprocesses input text:
    - Lowercase
    - Remove punctuation
    - Tokenize
    - Remove stopwords
    - Lemmatize

    Returns: cleaned string
    """
    if not isinstance(text, str):  # Handle NaN or non-str
        return ""

    text = text.lower()  # lowercase
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # remove punctuation
    tokens = tokenizer.tokenize(text)  # tokenize

    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]

    return " ".join(cleaned)
