from datasets import load_dataset
from transformers import DistilBertTokenizer, DataCollatorWithPadding, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import torch 
import evaluate
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report


# Load dataset directly using Hugging Face `datasets`
dataset = load_dataset("csv", data_files="/Users/sofie/.cache/kagglehub/datasets/mrmorj/hate-speech-and-offensive-language-dataset/versions/1/labeled_data.csv")

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

# Compute Class Weights (AFTER tokenization)
labels = np.array(tokenized_datasets["train"]["label"])  # Extract training labels
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)  # Convert to PyTorch tensor

# Reduce the weight for Hate Speech (Class 0) slightly to prevent overcompensation
class_weights[0] *= 0.75  # Reduce the weight by 25% (adjustable)

# Normalize class weights to keep relative proportions
class_weights /= class_weights.sum() * 3  # Ensures sum remains stable

# Initialize Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Convert logits to class predictions
    
    # Compute per-class metrics
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)

    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "precision_weighted": precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"],
        "recall_weighted": recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"],
        "f1_weighted": f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"],
        # Per-class metrics
        "precision_hate_speech": report["0"]["precision"],
        "recall_hate_speech": report["0"]["recall"],
        "f1_hate_speech": report["0"]["f1-score"],
        "precision_offensive": report["1"]["precision"],
        "recall_offensive": report["1"]["recall"],
        "f1_offensive": report["1"]["f1-score"],
        "precision_neutral": report["2"]["precision"],
        "recall_neutral": report["2"]["recall"],
        "f1_neutral": report["2"]["f1-score"],
    }

# # Define function to compute all metrics
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)  # Convert logits to class predictions
    
#     return {
#         "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
#         "precision": precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"],
#         "recall": recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"],
#         "f1": f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]}

# Load model
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # Accept extra arguments
        labels = inputs.pop("labels")  # Extract labels
        outputs = model(**inputs)  # Get model predictions
        logits = outputs.logits
        loss_fn = CrossEntropyLoss(weight=class_weights.to(logits.device))  # Apply class weights
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate at the end of every epoch
    save_strategy="epoch",
    load_best_model_at_end=True,  # Automatically load best model
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=7,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    compute_metrics=compute_metrics,  
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Add Early Stopping
)

# Train and evaluate model
trainer.train()

# Get predictions
predictions = trainer.predict(tokenized_datasets["test"]).predictions
predictions = np.argmax(predictions, axis=-1)  # Convert logits to class predictions

# Count occurrences of each class in predictions
unique, counts = np.unique(predictions, return_counts=True)
print("Predicted Class Counts:", dict(zip(unique, counts)))

# Count occurrences of each class in test labels
test_labels = tokenized_datasets["test"]["label"]
unique, counts = np.unique(test_labels.numpy(), return_counts=True)
print("Actual Class Counts:", dict(zip(unique, counts)))

trainer.evaluate()
