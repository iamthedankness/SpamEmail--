import pandas as pd
import chardet

# Detect encoding of the CSV file
with open('spam.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

# Load dataset with detected encoding
emails = pd.read_csv('spam.csv', encoding=encoding)

# Display the first few rows of the dataset
print(emails.head())
