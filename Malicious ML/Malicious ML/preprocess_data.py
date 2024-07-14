import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data['url'], data['type']

def preprocess_data(file_path):
    urls, labels = load_data(file_path)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(urls)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer
