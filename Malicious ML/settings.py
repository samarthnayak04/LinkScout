import os

# Define the directory and file structure
structure = {
    "Malicious ML": {
        "data": {
            "malware_urls.csv": "url,label\nexample.com,1\nexample.org,0"  # Sample data
        },
        "preprocess_data.py": """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data['url'], data['label']

def preprocess_data(file_path):
    urls, labels = load_data(file_path)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(urls)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer
""",
        "train_model.py": """from preprocess_data import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

data_dir = 'C:/Users/shake/OneDrive/Desktop/Malicious ML/data/malware_urls.csv'

X_train, X_test, y_train, y_test, vectorizer = preprocess_data(data_dir)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

joblib.dump(model, 'C:/Users/shake/OneDrive/Desktop/Malicious ML/model.joblib')
joblib.dump(vectorizer, 'C:/Users/shake/OneDrive/Desktop/Malicious ML/vectorizer.joblib')
""",
        "predict.py": """import joblib

model = joblib.load('C:/Users/shake/OneDrive/Desktop/Malicious ML/model.joblib')
vectorizer = joblib.load('C:/Users/shake/OneDrive/Desktop/Malicious ML/vectorizer.joblib')

def predict(url):
    X = vectorizer.transform([url])
    prediction = model.predict(X)
    return prediction[0]

if __name__ == "__main__":
    url = input("Enter a URL to check: ")
    prediction = predict(url)
    print(f"The URL is {'malicious' if prediction == 1 else 'benign'}")
""",
        "utils.py": """# Placeholder for utility functions

def example_utility_function():
    pass
"""
    }
}

# Function to create the directory structure
def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, 'w') as f:
                f.write(content)

# Create the structure
base_path = "."
create_structure(base_path, structure)
