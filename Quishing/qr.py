import joblib
import pandas as pd
from urllib.parse import urlparse
import scipy.sparse
import cv2
from pyzbar.pyzbar import decode

# Load the saved model, label encoder, and vectorizer
model = joblib.load('random_forest_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Function to extract features from URLs
def extract_url_features(url):
    parsed_url = urlparse(url)
    return {
        'domain': parsed_url.netloc,
        'path_length': len(parsed_url.path),
        'query_length': len(parsed_url.query),
        'num_subdomains': len(parsed_url.netloc.split('.')) - 1
    }

# Function to read QR code and extract URL
def read_qr_code(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is None:
        print(f"Error: Unable to open image file: {image_path}")
        return None

    # Decode QR code
    qr_codes = decode(image)

    # If QR code is detected, extract the data
    if qr_codes:
        qr_code = qr_codes[0]  # Assuming there is only one QR code in the image
        qr_data = qr_code.data.decode('utf-8')
        print(f"QR Code data: {qr_data}")
        return qr_data
    else:
        print("QR Code not detected")
        return None

# Main function
def main():
    # Ask the user to input the path to the QR code image
    image_path = input("Please enter the path to the QR code image: ").strip()

    # Strip quotes if present
    if (image_path.startswith('"') and image_path.endswith('"')) or (image_path.startswith("'") and image_path.endswith("'")):
        image_path = image_path[1:-1]

    # Extract the URL from the QR code
    test_url = read_qr_code(image_path)

    if test_url:
        # Extract features from the test URL
        url_features = extract_url_features(test_url)
        url_features_df = pd.json_normalize(url_features)

        # Vectorize the domain column
        domain_vectorized = vectorizer.transform([url_features['domain']])

        # Convert the sparse matrix to a DataFrame without converting to dense array
        domain_df = pd.DataFrame.sparse.from_spmatrix(domain_vectorized, columns=vectorizer.get_feature_names_out())

        # Combine extracted features with the domain features
        test_df = pd.concat([url_features_df, domain_df], axis=1)

        # Ensure all features are numerical and handle NaN values
        test_df = test_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Prepare the dataset for prediction
        X_test = scipy.sparse.hstack([scipy.sparse.csr_matrix(test_df.drop(columns=domain_df.columns).values), domain_vectorized])

        # Make prediction using the loaded model
        y_pred = model.predict(X_test)

        # Print the prediction
        print("Prediction:", y_pred[0])

if __name__ == '__main__':
    main()
