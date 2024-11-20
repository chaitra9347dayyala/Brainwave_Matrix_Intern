# Brainwave_Matrix_Intern
Project Overview: This Fake News Detection System uses machine learning techniques to automatically classify news articles as Fake News or True News. The system leverages two powerful models: Logistic Regression and Long Short-Term Memory (LSTM). These models analyze the content of news articles to predict whether they are fake or authentic.

System Components:

Dataset: The system uses two datasets: one containing fake news articles (Fake.csv) and another containing true news articles (True.csv). These datasets are pre-processed and combined into a single dataset, with a label (class column) indicating whether the news is fake (0) or true (1).

Preprocessing: The text data undergoes several preprocessing steps:

Lowercasing: All text is converted to lowercase.
Removing Punctuation & Special Characters: Unnecessary characters, such as punctuation, HTML tags, and digits, are removed.
Stopword Removal: Commonly used words like "the", "is", etc., are removed to focus on meaningful words.
URL Removal: Any links or references to URLs are stripped from the text.
Feature Extraction: The system uses TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization to convert text into numerical format, which makes it possible to apply machine learning models for classification.

Machine Learning Models:

Logistic Regression Model: A classical machine learning model that learns to classify news articles based on the features extracted from the text. The model is trained on a subset of the dataset and evaluated for its performance using various metrics such as accuracy, precision, recall, F1 score, and confusion matrix.

LSTM (Long Short-Term Memory) Model: A type of Recurrent Neural Network (RNN) specifically designed for sequence data like text. It captures the temporal dependencies between words in a news article and is particularly effective for analyzing long sequences of text.

Evaluation Metrics: The system evaluates both models using the following metrics:

Accuracy: Measures the overall correctness of the model's predictions.
Precision: Indicates how many of the predicted positive news articles were actually true.
Recall: Measures how many of the actual true news articles were correctly identified.
F1 Score: A balance between precision and recall.
Confusion Matrix: A visual representation of the model's performance, showing the true positive, true negative, false positive, and false negative predictions.
Manual Testing: To test the models manually, 20 rows from the dataset are set aside for evaluation. These rows are not included in the training and test datasets but can be used to verify the system’s performance with real examples.

Model Performance: Both models are trained and evaluated, with accuracy scores and other performance metrics displayed. The LSTM model, due to its deep learning nature, is expected to perform better on text data compared to the simpler Logistic Regression model.

Conclusion: This system provides an automated solution for classifying news articles into fake or true categories. By using advanced machine learning and natural language processing techniques, it offers an efficient tool for combating misinformation in the media.

