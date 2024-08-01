import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2

# Load the dataset
file_path = 'spam.csv'
emails = pd.read_csv(file_path, encoding='latin-1')

# Rename columns to match the dataset structure
emails.columns = ['label', 'text', 'unused1', 'unused2', 'unused3']

# Drop unused columns
emails = emails[['label', 'text']]

# Pre-processing
emails['text'] = emails['text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
emails.dropna(inplace=True)

# Verify class distribution
print("Class distribution in the entire dataset:")
print(emails['label'].value_counts())

# Feature Extraction
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(emails['text'])
y = emails['label']

# Feature Selection
selector = SelectKBest(chi2, k=500)  # Adjust k based on your requirement
X = selector.fit_transform(X, y)

# Split data ensuring both classes are represented
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Check class distribution in both sets
def check_class_distribution(y_train, y_test):
    print("\nClass distribution in training set:")
    print(y_train.value_counts())
    print("\nClass distribution in test set:")
    print(y_test.value_counts())

check_class_distribution(y_train, y_test)

# Initialize models
models = {
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Support Vector Machine': SVC(kernel='linear', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

# Function to evaluate and print metrics for each model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    labels = y_train.unique()
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy
    report = classification_report(y_test, y_pred, labels=labels)
    
    return conf_matrix, accuracy, error_rate, report, training_time, prediction_time

# Test and evaluate each model
for name, model in models.items():
    print(f"\nTesting {name}...")
    try:
        conf_matrix, accuracy, error_rate, report, training_time, prediction_time = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(f"Confusion Matrix for {name}:\n{conf_matrix}")
        print(f"Accuracy for {name}: {accuracy:.4f}")
        print(f"Error Rate for {name}: {error_rate:.4f}")
        print(f"Classification Report for {name}:\n{report}")
        print(f"Training Time for {name}: {training_time:.4f} seconds")
        print(f"Prediction Time for {name}: {prediction_time:.4f} seconds")
    except ValueError as e:
        print(f"Error testing {name}: {e}")
    print("-" * 80)
