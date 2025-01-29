import pandas as pd
import tensorflow as tf
import transformers
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from datasets import Dataset

file_path = "/Users/sofie/.cache/kagglehub/datasets/mrmorj/hate-speech-and-offensive-language-dataset/versions/1/labeled_data.csv"

# Import data from csv file using path
data = pd.read_csv(file_path)

# Assign features (X) and labels (y)
X = data["tweet"]  # Feature: the tweets 
y = data["class"]  # Label: the classes (0: hate speech, 1: offensive language, 2: neither)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

# # Create training and testing DataFrames
# train_df = pd.DataFrame({"text": train_X, "label": train_y})
# test_df = pd.DataFrame({"text": test_X, "label": test_y})

# # Convert to Hugging Face Dataset
# train_ds = Dataset.from_pandas(train_df)
# test_ds = Dataset.from_pandas(test_df)


model_name = "distilbert-base-uncased"
model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
classes = ["hate speech", "offensive language", "neither"]

# Tokenization function with padding & truncation
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Apply tokenization to the datasets
train_ds = train_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)