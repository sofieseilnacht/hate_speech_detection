from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "/Users/sofie/.cache/kagglehub/datasets/mrmorj/hate-speech-and-offensive-language-dataset/versions/1/labeled_data.csv"
data = pd.read_csv(file_path)

# Assign features & labels
X = data["tweet"][:100]
y = data["class"][:100]

# Split into train & validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Load tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Tokenize dataset
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128, return_tensors="pt")
val_encodings = tokenizer(list(X_val), truncation=True, padding=True, max_length=128, return_tensors="pt")

# Convert labels to PyTorch tensors
y_train = torch.tensor(y_train.values)
y_val = torch.tensor(y_val.values)

# Load model in PyTorch
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)
