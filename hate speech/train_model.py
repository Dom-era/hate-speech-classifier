import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv('dataset.csv')

# Prepare the features and labels
X = data['text']
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CountVectorizer
vectorizer = CountVectorizer()

# Transform the training data
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Save the trained model and vectorizer
joblib.dump(model, 'hate_speech_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer trained and saved successfully!")
