import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset

file_path = "/Users/sofie/.cache/kagglehub/datasets/mrmorj/hate-speech-and-offensive-language-dataset/versions/1/labeled_data.csv"

# Import data from csv file using path
data = pd.read_csv(file_path)

# Assign features (X) and labels (y)
X = data["tweet"]  # Feature: the tweets 
y = data["class"]  # Label: the classes (0: hate speech, 1: offensive language, 2: neither)

# Cut the dataset a bit so we can fine tune properly
small_data = pd.DataFrame({"tweet": X, "class": y})

classes = ["hate speech", "offensive language", "neither"]

model_name = "distilbert-base-uncased"
model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

def classify_tweet(tweet):
    # Tokenize input (directly return tensors)
    inputs = tokenizer(tweet, truncation=True, padding="max_length", max_length=128, return_tensors="tf")

    # Get model predictions (logits)
    logits = model(inputs)[0]

    # Convert logits to probabilities using softmax
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]

    # Get the predicted class
    predicted_class = np.argmax(probs)

    print(f"Tweet: {tweet}")
    print(f"Predicted class: {classes[predicted_class]}")
    
    return predicted_class

# # Apply classification only on the reduced dataset
# small_data["predicted_class"] = small_data["tweet"].apply(classify_tweet)

# # Print results
# print(small_data[["tweet", "class", "predicted_class"]].head(10))

# # Compute accuracy only on the reduced dataset
# accuracy = accuracy_score(small_data["class"], small_data["predicted_class"])
# print(f"Model Accuracy: {accuracy * 100:.2f}%")

# # Show misclassified examples
# misclassified = small_data[small_data["class"] != small_data["predicted_class"]]
# print(misclassified[["tweet", "class", "predicted_class"]].head(10))