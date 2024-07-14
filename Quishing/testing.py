# import joblib
# import pandas as pd
# from urllib.parse import urlparse
# import scipy.sparse

# # Load the saved model, label encoder, and vectorizer
# model = joblib.load('random_forest_model.joblib')
# label_encoder = joblib.load('label_encoder.joblib')
# vectorizer = joblib.load('vectorizer.joblib')

# # Function to extract features from URLs
# def extract_url_features(url):
#     parsed_url = urlparse(url)
#     return {
#         'domain': parsed_url.netloc,
#         'path_length': len(parsed_url.path),
#         'query_length': len(parsed_url.query),
#         'num_subdomains': len(parsed_url.netloc.split('.')) - 1
#     }

# # Example test URL
# test_url = "games.teamxbox.com/xbox-360/1189/Condemned-Criminal-Origins/"
# # Extract features from the test URL
# url_features = extract_url_features(test_url)
# url_features_df = pd.json_normalize(url_features)

# # Vectorize the domain column
# domain_vectorized = vectorizer.transform([url_features['domain']])

# # Convert the sparse matrix to a DataFrame without converting to dense array
# domain_df = pd.DataFrame.sparse.from_spmatrix(domain_vectorized, columns=vectorizer.get_feature_names_out())

# # Combine extracted features with the domain features
# test_df = pd.concat([url_features_df, domain_df], axis=1)

# # Ensure all features are numerical and handle NaN values
# test_df = test_df.apply(pd.to_numeric, errors='coerce').fillna(0)

# # Prepare the dataset for prediction
# X_test = scipy.sparse.hstack([scipy.sparse.csr_matrix(test_df.drop(columns=domain_df.columns).values), domain_vectorized])

# # Make prediction using the loaded model
# y_pred = model.predict(X_test)

# # Print the prediction
# print("Prediction:", y_pred[0])



import joblib
import pandas as pd
from urllib.parse import urlparse
import scipy.sparse

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

# Ask the user to input a URL
test_url = input("Please enter the URL to be analyzed: ")

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
