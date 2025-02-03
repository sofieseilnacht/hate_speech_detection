from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset  # Hugging Face Datasets

# Load dataset
file_path = "/Users/sofie/.cache/kagglehub/datasets/mrmorj/hate-speech-and-offensive-language-dataset/versions/1/labeled_data.csv"
data = pd.read_csv(file_path)

# Assign features & labels
X = data["tweet"][:100]
y = data["class"][:100]

# First, split into train (70%) and temp (30%) = (validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Now, split temp into validation (50%) and test (50%) (15% each of full dataset)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# Tokenize dataset using Hugging Face `Dataset`
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Convert to Hugging Face Dataset format
train_data = Dataset.from_dict({"text": list(X_train), "labels": list(y_train)})
val_data = Dataset.from_dict({"text": list(X_val), "labels": list(y_val)})
test_data = Dataset.from_dict({"text": list(X_test), "labels": list(y_test)})

# Apply tokenization
train_dataset = train_data.map(tokenize_function, batched=True)
val_dataset = val_data.map(tokenize_function, batched=True)
test_dataset = test_data.map(tokenize_function, batched=True)


# Load model
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize Trainer with simplified dataset handling
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("hate_speech_pytorch")
tokenizer.save_pretrained("hate_speech_pytorch")
