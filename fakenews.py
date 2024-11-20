import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
import string
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load the datasets
df_fake = pd.read_csv(r"C:\Users\DELL\Documents\New folder\dataset\Fake.csv")
df_true = pd.read_csv(r"C:\Users\DELL\Documents\New folder\dataset\True.csv")

# Add a 'class' column to indicate fake (0) and true (1)
df_fake['class'] = 0  # Fake news as 0
df_true['class'] = 1  # True news as 1

# Merge the datasets
df = pd.concat([df_fake, df_true], ignore_index=True)

# Remove unwanted columns (if necessary) and duplicate rows
df = df[['title', 'text', 'class']].drop_duplicates(subset='text')

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Remove 20 rows from the end for manual testing
manual_testing_data = df.iloc[-20:]
df = df.iloc[:-20]

# Preprocessing function: Lowercase, remove punctuation, remove stopwords
def wordopt(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub('\[.*?\]', '', text)  # Remove anything in square brackets
    text = re.sub("\\W", " ", text)  # Remove non-word characters
    text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub('<.*?>+', '', text)  # Remove HTML tags
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub('\n', '', text)  # Remove new lines
    text = re.sub('\w*\d\w*', '', text)  # Remove digits
    return text

# Apply preprocessing to the text column
df['text'] = df['text'].apply(wordopt)

# Split data into independent (X) and dependent (y) variables
X = df['text']
y = df['class']

# Convert text data into numerical data using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Logistic Regression Model
print("Logistic Regression Evaluation:")

# Initialize and train the Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Make predictions
lr_predictions = lr_model.predict(X_test)

# Calculate performance metrics for Logistic Regression
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print("Precision:", precision_score(y_test, lr_predictions))
print("Recall:", recall_score(y_test, lr_predictions))
print("F1 Score:", f1_score(y_test, lr_predictions))
print("Classification Report:\n", classification_report(y_test, lr_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_predictions))


# Preprocess the data for LSTM
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X)

# Convert text to sequences and pad them to ensure uniform length
X_seq = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_seq, maxlen=100, padding='post', truncating='post')

# Split the data into training and testing sets (80% train, 20% test)
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# LSTM Model
print("\nLSTM Model Evaluation:")

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
lstm_model.add(LSTM(128))
lstm_model.add(Dense(1, activation='sigmoid'))

# Compile and train the model
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, verbose=1)

# Make predictions using the LSTM model
lstm_predictions = (lstm_model.predict(X_test_lstm) > 0.5).astype("int32")

# Calculate performance metrics for LSTM
print("Accuracy:", accuracy_score(y_test_lstm, lstm_predictions))
print("Precision:", precision_score(y_test_lstm, lstm_predictions))
print("Recall:", recall_score(y_test_lstm, lstm_predictions))
print("F1 Score:", f1_score(y_test_lstm, lstm_predictions))
print("Classification Report:\n", classification_report(y_test_lstm, lstm_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test_lstm, lstm_predictions))

# Output for manual testing
print("\nManual Testing Data (Last 20 rows):")
print(manual_testing_data[['text', 'class']])
