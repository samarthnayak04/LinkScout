import pandas as pd

# Load the dataset without extra quotes around the file path
df = pd.read_csv(r"C:\MalwareTotal\Kaggle\tessst.csv")

# Display basic information about the dataset
print(df.info())
print(df.head())


# Check for missing values
print(df.isnull().sum())

# Remove rows with missing URLs
df = df.dropna(subset=['url'])

# Fill or drop other missing values based on your data strategy
# Example: Fill missing numerical values with the median
df = df.fillna(df.median(numeric_only=True))

# Check again for missing values
print(df.isnull().sum())


# Check for duplicate URLs
print("Number of duplicate URLs:", df['url'].duplicated().sum())

# Remove duplicate URLs
df = df.drop_duplicates(subset=['url'])

# Verify removal
print("Number of duplicate URLs after removal:", df['url'].duplicated().sum())

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to extract URL components
def extract_url_components(url):
    url_components = re.findall(r'\w+\.\w+', url)  # Adjust regex as needed
    return ' '.join(url_components)

# Load data from CSV file
input_csv_path = r"C:\MalwareTotal\Kaggle\tessst.csv"  # Replace with your CSV file path
output_csv_path = r"C:\MalwareTotal\Kaggle\output.csv"  # Replace with desired output CSV file path

# Read the CSV into a DataFrame
df = pd.read_csv(input_csv_path)

# Apply URL component extraction
df['url_components'] = df['url'].apply(extract_url_components)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
tfidf_matrix = vectorizer.fit_transform(df['url_components'])

# Convert TF-IDF matrix to DataFrame (sparse matrix version)
tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix, columns=vectorizer.get_feature_names_out())

# Concatenate TF-IDF features with original DataFrame
df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

# Save the processed data to a new CSV file
df.to_csv(output_csv_path, index=False)

print(f"Processed data saved to {output_csv_path}")


