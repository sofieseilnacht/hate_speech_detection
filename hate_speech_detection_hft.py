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
import keras
from tensorflow.keras.optimizers import Adam  # No legacy needed

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

# Tokenize training & validation sets
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128, return_tensors="tf")
val_encodings = tokenizer(list(X_val), truncation=True, padding=True, max_length=128, return_tensors="tf")

# Convert labels to TensorFlow tensors
y_train = tf.convert_to_tensor(y_train.values)
y_val = tf.convert_to_tensor(y_val.values)

# Create TensorFlow datasets (batch processing)
batch_size = 8  # Adjust based on available GPU/CPU memory
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).shuffle(len(X_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), y_val)).batch(batch_size)

# Load pre-trained DistilBERT model
model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Fix Variable Initialization Issue
dummy_input = tf.zeros((1, 128), dtype=tf.int32)  # Force initialization
model(dummy_input)

# Define additional evaluation metrics
METRICS = [
    keras.metrics.TruePositives(name='tp'), 
    keras.metrics.FalsePositives(name='fp'), 
    keras.metrics.TrueNegatives(name='tn'), 
    keras.metrics.FalseNegatives(name='fn'),  
    keras.metrics.Precision(name='precision'), 
    keras.metrics.Recall(name='recall'), 
    keras.metrics.CategoricalAccuracy(name='acc'),
    keras.metrics.AUC(name='auc'),
]

# Compile model using the legacy Adam optimizer
model.compile(
    optimizer=Adam(learning_rate=5e-5), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=METRICS
)

# Fine-tune the model 🚀
epochs = 3  # Increase if needed for better performance
model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# Save the trained model
model.save_pretrained("hate_speech_model")
tokenizer.save_pretrained("hate_speech_model")

print("✅ Model fine-tuned and saved successfully!")
