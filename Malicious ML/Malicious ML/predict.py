import joblib

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
