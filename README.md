# Hate Speech Detection Using Hugging Face Transformers

The Hate Speech Detection project aims to build an automated system that can classify text as hate speech, offensive speech, or neutral language using Natural Language Processing (NLP) and Machine Learning (ML). The goal is to develop a robust classifier capable of distinguishing harmful content from general discourse, making online platforms safer by identifying and mitigating toxic language.

## Overview

The purpose of this script is to:

- Train an ML model to classify text into predefined categories (Hate Speech, Offensive Language, Neutral).
- Apply NLP techniques such as tokenization, class reweighting, and word embeddings to improve classification accuracy.
- Experiment with Hugging Face models, including DistilBERT, to optimize text processing and classification.
- Evaluate model performance using key metrics: precision, recall, F1-score, and accuracy.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Dependencies](#dependencies)
4. [Dataset](#dataset) 
5. [Class Imbalance](#class-imbalance)
6. [How It Works](#how-it-works)
7. [Data Preprocessing](#data-preprocessing)
8. [Model Training & Optimization](#model-training-&-optimization)
9. [Evaluation](#evaluation)
10. [Challenges in Hate Speech Classification](#challenges-in-hate-speech-classification)
11. [Future Improvements](#future-improvements)

# Getting Started

To run this code, you'll need Python 3.12.4 installed and to set up the required dependencies.

## Dependencies

The following Python libraries are required:

- Hugging Face Transformers (transformers, datasets, and evaluate libraries)
- scikit-learn
- NumPy
- Pytorch

## Dataset

The dataset, "Hate Speech and Offensive Language Data", was sourced from Kaggle and consists of 24,783 labeled tweets categorized into three classes:

- 0 – Hate Speech
- 1 – Offensive Language
- 2 – Neutral Speech

The raw data includes seven columns:

- tweet index - For each tweet, a unique identifier that serves as its position within the dataset
- count – Number of CrowdFlower annotators who labeled each tweet
- hate speech – Number of annotators who marked the tweet as hate speech
- offensive language – Number of annotators who marked it as offensive language
- neither – Number of annotators who judged the tweet as neutral
- class – Final class label based on the majority vote of annotators (0, 1, or 2)
- tweet – The raw text of the tweet

## Class Imbalance

The dataset is highly imbalanced:

- The majority of tweets are labeled offensive language (1).
- Neutral tweets (2) are much less frequent.
- Hate speech tweets (0) are significantly underrepresented.

# How It Works

## Data Preprocessing

Using the Hugging Face Transformers (HFT) datasets library, we performed the following preprocessing steps:

- Loaded and cleaned the dataset, removing unnecessary columns and renaming the remaining two to match HFT's expected format: "label" (formerly "class") and "text" (formerly "tweet").
- Split the dataset into training (80%), validation (10%), and test (10%) sets.
- Tokenized and padded text sequences, ensuring compatibility with the model input format. Any text exceeding the model's maximum token length was truncated.

To address class imbalance, we reweighted the class distribution using the PyTorch library and cross-entropy loss function, assigning higher weights to the minority class (label 0) to balance representation. We also replaced DistilBERT’s default loss function with this reweighted loss to improve model performance on underrepresented classes.

## Model Training & Optimization:

We fine-tuned the DistilBERT transformer model using Hugging Face's transformers library, experimenting with different epochs, learning rates, and class weight adjustments. We evaluated each model based on performance metrics and selected the best-performing model.
To ensure generalization, we tested the model on unseen data at two stages:

- Validation Data: During training, to monitor improvements and prevent overfitting.
- Test Data: After training, to assess real-world performance.

The best model was trained using:

- Class Rebalancing: Increased the 0-label class weight to twice the default to address imbalance.
- Training Epochs: 7 epochs with early stopping and best model callbacks.
- Learning Rate: 3e-5 (adjusted for optimal convergence).

## Evaluation:

Using precision, recall, and F1-score for the overall model perfomance as well as each of the classes to assess model performance, we obtained the following metrics for our best model:

|  Hate Speech (0)  | Offensive Language (1) |    Neither (2)   | 
|-------------------|------------------------|------------------|
| Precision: 0.608  | Precision: 0.932       | Precision: 0.904 |              
| Recall: 0.333     | Recall: 0.969          | Recall: 0.860    |              
| F1-Score: 0.431   | F1-Score: 0.951        | F1-Score: 0.882  |     

|       Overall Model       | 
|---------------------------|
| Accuracy: 91.8%           |               
| Weighted F1-Score: 91.2%  |               

# Challenges in Hate Speech Classification

Hate speech classification is inherently more challenging than distinguishing offensive vs. neutral language due to several key factors:

#### Context Dependence:

- Hate speech often relies on specific context, making it difficult for NLP models to differentiate between harmful speech and general discourse.
- Subtle variations in phrasing can completely change the meaning.

#### Detection of Sarcasm & Irony:

- NLP models struggle with sarcasm and irony, which are common in online discussions.
- A phrase may appear neutral or even positive in isolation but convey hostility when interpreted within a broader context.

#### Class Imbalance & Label Ambiguity:

- Hate speech is often underrepresented in datasets, leading to imbalanced class distributions.
- Some instances may fall between offensive and hate speech, making it difficult for both annotators and models to assign a definitive label.

#### Challenges with Coded & Subtle Hate Speech:

- Coded language (e.g., alternative spellings, euphemisms) is commonly used to evade moderation.
- Subtle hate speech, which may use dog whistles or implicit biases, is harder for models to recognize compared to explicit slurs.

Because of these factors, improving classification performance requires advanced NLP techniques, such as context-aware embeddings, transformer-based models, and bias-aware training strategies.

# Future Improvements
 
Since recall for hate speech is still low, we can explore several approaches to improve our model’s performance. One potential solution is to slightly lower the class weight for hate speech. Currently, hate speech is heavily weighted, making the model more cautious in its predictions. Reducing the weight slightly, for example, by multiplying it by 0.8 or 0.9, could allow for greater flexibility. However, adjusting the weight alone may reintroduce issues with class imbalance, which is why additional strategies should be considered. Another approach is to increase data augmentation for hate speech or obtain more labeled hate speech data to improve training. Because hate speech is the rarest category, augmenting the dataset using techniques like paraphrasing, back translation, or synonym replacement could enhance model robustness. Additionally, collecting more real-world hate speech examples from sources such as Twitter, Instagram, and Facebook could further strengthen the dataset.

Beyond data augmentation, we can also experiment with transfer learning by utilizing alternative transformer models such as RoBERTa or HateBERT instead of DistilBERT. DistilBERT is designed for efficiency rather than state-of-the-art accuracy, which may limit performance in this classification task. If computational resources allow, fine-tuning a more robust model like RoBERTa (roberta-base) or HateBERT, which has been specifically trained on hate speech detection, could lead to improvements in recall.

Finally, a detailed analysis of misclassified examples could provide insights into specific failure cases. By examining instances where the model misclassifies hate speech, we can determine whether certain factors contribute to errors, such as longer sentences, misclassification of specific words, or difficulties distinguishing between subtle and explicit hate speech. Understanding these patterns can guide further refinements to preprocessing techniques, model architecture, or training strategies.