
import pandas as pd
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import scipy.sparse
import joblib

# Load the enhanced dataset
df = pd.read_csv('enhanced_cybersecurity_dataset.csv')

# Function to extract features from URLsC:\MalwareTotal\Kaggle\malicious_phish.csv
def extract_url_features(url):
    parsed_url = urlparse(url)
    return {
        'domain': parsed_url.netloc,
        'path_length': len(parsed_url.path),
        'query_length': len(parsed_url.query),
        'num_subdomains': len(parsed_url.netloc.split('.')) - 1
    }

# Apply the feature extraction function
url_features = df['url'].apply(extract_url_features)
url_features_df = pd.json_normalize(url_features)

# Combine extracted features with the original dataframe
df = pd.concat([df, url_features_df], axis=1)

# Drop the original URL column as it's no longer needed
df.drop('url', axis=1, inplace=True)

# Encode the 'type' column
label_encoder = LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])

# Vectorize the domain column with limited features
vectorizer = TfidfVectorizer(max_features=1000)  # Limit to 1000 features
domain_vectorized = vectorizer.fit_transform(df['domain'])

# Convert the sparse matrix to a DataFrame without converting to dense array
domain_df = pd.DataFrame.sparse.from_spmatrix(domain_vectorized, columns=vectorizer.get_feature_names_out())

# Combine with the main dataframe
df = pd.concat([df, domain_df], axis=1)

# Drop the original domain column as it's no longer needed
df.drop('domain', axis=1, inplace=True)

# Prepare the dataset
X = df.drop('label', axis=1)  # Features  8============D
y = df['label']               # Target variable 8====D

# Convert X to a sparse matrix to ensure it remains sparse
X_sparse = scipy.sparse.hstack([scipy.sparse.csr_matrix(X.drop(columns=domain_df.columns)), domain_vectorized])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sparse, y, test_size=0.2, random_state=42)

# Model Development
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate Model Performance
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model, label encoder, and vectorizer
joblib.dump(model, 'random_forest_model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
