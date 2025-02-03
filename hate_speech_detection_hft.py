from datasets import load_dataset
from transformers import DistilBertTokenizer, DataCollatorWithPadding, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch 


# Load dataset directly using Hugging Face `datasets`
dataset = load_dataset("csv", data_files="/Users/sofie/.cache/kagglehub/datasets/mrmorj/hate-speech-and-offensive-language-dataset/versions/1/labeled_data_copy.csv")

# Keep only 'text' and 'labels' columns
dataset = dataset.remove_columns(["Unnamed: 0", "count", "hate_speech", "offensive_language", "neither"])

# Rename columns to match what Hugging Face expects
dataset = dataset["train"].rename_column("tweet", "text").rename_column("class", "label")

# First, split into 80% train and 20% temp (which will be further split into val + test)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Now, split temp into 50% validation and 50% test (10% each of full dataset)
temp = dataset["test"].train_test_split(test_size=0.5, seed=42)

# Assign new splits correctly
dataset["val"] = temp["train"]  # 10% Validation Set
dataset["test"] = temp["test"]  # 10% Test Set

# Load tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Remove the original "text" column (model doesn't need it since we're using encoded data now rather than raw text)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# Convert dataset to PyTorch format
tokenized_datasets.set_format("torch")

# Initialize Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print(tokenized_datasets)
# # Apply tokenization
# dataset = dataset.map(tokenize_function, batched=True)

# # Remove unnecessary columns
# dataset = dataset.remove_columns(["text"])  # Keep only tokenized features and labels

# # Convert dataset to PyTorch format
# dataset.set_format("torch")

# # Extract train, validation, and test datasets
# train_dataset = dataset["train"]
# val_dataset = dataset["test"]["train"]
# test_dataset = dataset["test"]["test"]

# # Load model
# model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_dir="./logs",
# )

# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # Train the model
# trainer.train()

# # Save the fine-tuned model
# model.save_pretrained("hate_speech_pytorch")
# tokenizer.save_pretrained("hate_speech_pytorch")

# print("✅ Model fine-tuned and saved successfully!")
