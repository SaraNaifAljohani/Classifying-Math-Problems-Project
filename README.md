# 🧠 Math Problem Classifier: Fine-Tuning BERT for NLP

## 🚀 Project Overview

This project implements a **Natural Language Processing (NLP) classification system** designed to automatically categorize math word problems into one of **8 distinct academic categories** (e.g., Algebra, Geometry, Arithmetic). 

This is for my participation for [KAChallenges Series: Classifying Math Problems](https://www.kaggle.com/competitions/classification-of-math-problems-by-kasut-academy) by KAUST Academy on Kaggle.

The solution utilizes **transfer learning** by **fine-tuning the `bert-base-uncased` transformer model** on a custom dataset, demonstrating a complete end-to-end machine learning pipeline from data preparation to model deployment and submission.

-----

## ✨ Key Features & Results

| Feature | Description |
| :--- | :--- |
| **Model Architecture** | Fine-tuned **`bert-base-uncased`** for multi-class sequence classification. |
| **Classification Task** | 8-class text classification (e.g., Algebra, Geometry, Statistics). |
| **Final Validation Accuracy** | Achieved **\~82.38% accuracy** (Notebook output) on the validation set. |
| **Maturity** | Complete pipeline including preprocessing, custom metric calculation, training, and final submission generation. |
| **Frameworks** | Built entirely within the **HuggingFace ecosystem** (Transformers, Datasets). |

-----

## 🛠️ Technical Stack & Technologies

| Category | Tools |
| :--- | :--- |
| **Language** | Python |
| **Deep Learning** | PyTorch, HuggingFace Transformers |
| **Data Handling** | Pandas, HuggingFace Datasets |
| **Machine Learning**| scikit-learn (for Label Encoding & Metrics) |
| **Environment** | Google Colab (using T4 GPU acceleration) |

-----

## 💻 Methodology & Pipeline

The project follows a rigorous NLP and deep learning workflow:

1.  **Data Preparation:**
      * Loaded the `train.csv` data and used **`sklearn.preprocessing.LabelEncoder`** to convert the categorical problem types (labels) into numerical IDs.
      * Split the data into 80% training and 20% validation sets.
2.  **Tokenization:**
      * Utilized the **`BertTokenizer`** to convert raw text math questions into input IDs and attention masks, ensuring uniform length via padding and truncation.
3.  **Model Initialization:**
      * Loaded **`BertForSequenceClassification.from_pretrained("bert-base-uncased")`** and modified its final classification layer to predict the 8 output classes.
4.  **Training & Evaluation:**
      * Used the HuggingFace **`Trainer` API** for efficient training and GPU utilization (3 epochs).
      * Defined a **custom `compute_metrics` function** using **`accuracy_score`** from scikit-learn to track performance.
      * Final evaluation yielded a validation accuracy of **$82.38\%$**.
5.  **Submission:**
      * Generated predictions on the `test.csv` dataset and compiled the final results into the required `submission.csv` format for the Kaggle competition.

-----

## 🚀 Future Enhancements

  * **Hyperparameter Tuning:** Systematically test learning rates, batch sizes, and weight decay to maximize performance.
  * **Alternative Models:** Experiment with larger BERT variants (e.g., `bert-large`) or other models like RoBERTa or XLNet.
  * **Advanced Metrics:** Incorporate F1-Score, Precision, and Recall into the `compute_metrics` function for a more detailed analysis of class performance.