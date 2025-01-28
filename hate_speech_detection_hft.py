# import kagglehub
import pandas as pd
import os
import transformers
from transformers import DistilBertConfig, DistilBertTokenizer, TFDistilBertForSequenceClassification
import matplotlib.pyplot as plt

file_path = "/Users/sofie/.cache/kagglehub/datasets/mrmorj/hate-speech-and-offensive-language-dataset/versions/1/labeled_data.csv"
# print("Patgit h to dataset files:", file_path)

# Check if the path exists
# if os.path.exists(file_path):
#     # List all files in the directory
#     print("Files in the dataset directory:", os.listdir(file_path))
# else:
#     print("Dataset directory does not exist.")

# model_name = "distilbert-base-uncased"
# model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)
# tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Import data from csv file using path
data = pd.read_csv(file_path)

print(data.columns)

# Check for missing values in each column
missing_values = data.isnull().sum()
# Filter columns with NaN values
columns_with_nans = missing_values[missing_values > 0].index
# print(columns_with_nans)

# Access values from a specific column, e.g., 'column_name'
hate_speech = 'hate_speech'  # Replace with the column name you want to plot
hate_speech_values = data[hate_speech]

plt.plot(hate_speech_values)