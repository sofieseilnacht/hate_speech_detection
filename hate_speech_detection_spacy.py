import kagglehub
import pandas as pd
import os

file_path = "/Users/sofie/.cache/kagglehub/datasets/mrmorj/hate-speech-and-offensive-language-dataset/versions/1/labeled_data.csv"
print("Path to dataset files:", file_path)


data = pd.read_csv(file_path)
# Check if the path exists
# if os.path.exists(file_path):
#     # List all files in the directory
#     print("Files in the dataset directory:", os.listdir(file_path))
# else:
#     print("Dataset directory does not exist.")