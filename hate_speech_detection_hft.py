# import kagglehub
import pandas as pd
import os
import transformers
from transformers import DistilBertConfig, DistilBertTokenizer, TFDistilBertForSequenceClassification
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from datasets import Dataset

file_path = "/Users/sofie/.cache/kagglehub/datasets/mrmorj/hate-speech-and-offensive-language-dataset/versions/1/labeled_data.csv"

# Import data from csv file using path
data = pd.read_csv(file_path)

# Assign features (X) and labels (y)
X = data["tweet"]  # Feature: the text data
y = data["class"]  # Label: the target classes

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

# Create training and testing DataFrames
train_df = pd.DataFrame({"text": train_X, "label": train_y})
test_df = pd.DataFrame({"text": test_X, "label": test_y})

# Convert to Hugging Face Dataset
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# plt.plot(hate_speech_values)
# plt.title("Hate Speech Values")
# plt.xlabel("Index")
# plt.ylabel("Values")
# plt.show()





# model_name = "distilbert-base-uncased"
# model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)
# tokenizer = DistilBertTokenizer.from_pretrained(model_name)
