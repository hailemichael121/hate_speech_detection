import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def handle_missing_values(df, text_column='tweet'):
    if text_column in df.columns:
        df[text_column].fillna('', inplace=True)
    else:
        raise KeyError(f"Column '{text_column}' not found in DataFrame")
    return df

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(tokens)

def standardize_columns(df, text_column='tweet', label_column='class'):
    if text_column in df.columns and label_column in df.columns:
        df.rename(columns={text_column: 'text', label_column: 'label'}, inplace=True)
    else:
        raise KeyError(f"Columns '{text_column}' or '{label_column}' not found in DataFrame")
    return df[['text', 'label']]

def remove_outliers(df, column, max_length):
    df = df[df[column].apply(lambda x: len(x.split()) <= max_length)]
    return df
