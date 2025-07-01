import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
except FileNotFoundError:
    print("Error: spam.csv file not found. Please ensure the file is in your working directory.")
    print("You can download it from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
    print("\nCreating sample data for demonstration...")

    sample_data = {
        'v1': ['ham', 'spam', 'ham', 'spam', 'ham'] * 100,
        'v2': ['Hello how are you', 'Win money now call free', 'See you tomorrow',
               'Urgent call now win cash', 'Thanks for the help'] * 100
    }
    df = pd.DataFrame(sample_data)

# Rename columns
df.columns = ['label', 'text']

print(f"\n✅ Dataset shape: {df.shape}")
print("\n📊 Label distribution:")
print(df['label'].value_counts())
print("\n🔍 First few rows:")
print(df.head())

# Normalize text
df['text'] = df['text'].str.lower()
print("\n🔡 Text normalized to lowercase.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.3,
    random_state=42,
    stratify=df['label']
)

print(f"\n📦 Training set size: {len(X_train)}")
print(f"🧪 Test set size: {len(X_test)}")

# Vectorize
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"\n🧠 Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"📐 Training features shape: {X_train_vec.shape}")
print(f"📐 Test features shape: {X_test_vec.shape}")

# Train model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Predict
y_pred = nb_model.predict(X_test_vec)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📈 Classification Report:")
print("=" * 50)
report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])
print(report)

print("\n🧮 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


import joblib

# Save (dump) the model and vectorizer
joblib.dump(nb_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n💾 Model and vectorizer saved successfully.")


