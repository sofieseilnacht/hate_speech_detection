# import kagglehub
import pandas as pd
import os
import transformers
from transformers import DistilBertConfig, DistilBertTokenizer, TFDistilBertForSequenceClassification
import matplotlib.pyplot as plt

plt.ion()

file_path = "/Users/sofie/.cache/kagglehub/datasets/mrmorj/hate-speech-and-offensive-language-dataset/versions/1/labeled_data.csv"

# Import data from csv file using path
data = pd.read_csv(file_path)
print(data.columns)

# Access values from a specific column, e.g., 'column_name'
hate_speech = 'hate_speech'  # Replace with the column name you want to plot
hate_speech_values = data[hate_speech]

plt.plot(hate_speech_values)
# Show the plot
plt.title("Hate Speech Values")
plt.xlabel("Index")
plt.ylabel("Values")
plt.show()





# model_name = "distilbert-base-uncased"
# model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)
# tokenizer = DistilBertTokenizer.from_pretrained(model_name)
