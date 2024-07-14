import pandas as pd
from preprocess_data import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def main():
    data_dir = 'malicious_urls.csv'

    # Load and preprocess data
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(data_dir)

    # Initialize and train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    # Save model and vectorizer
    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')

if __name__ == "__main__":
    main()
