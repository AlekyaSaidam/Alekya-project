Advanced Machine Learning Approaches for Asthma Prediction Using XGBoost, SVM, and ANN Models
MSc Data Science Final Project
University of Hertfordshire
Module Code: 7PAM2002
Department: Physics, Astronomy and Mathematics
Student: Saidam. Alekya
SRN: 23036213
Supervisor: Carolyn Devereux
Date Submitted: 29 April 2025
Word Count: 5000
GitHub Repository: https://github.com/AlekyaSaidam/Alekya-project.git

Overview
This project investigates predictive models for asthma diagnosis using clinical and environmental health data. Three machine learning classifiers—XGBoost, Support Vector Machine (SVM), and Artificial Neural Network (ANN)—were evaluated individually and as a soft-voting ensemble. The objective was to enhance sensitivity and generalization for early asthma detection.

Objectives
Clean, preprocess, and analyze an asthma dataset

Train individual machine learning models: XGBoost, SVM, and ANN

Apply SMOTE to correct class imbalance

Evaluate models using F1-score, recall, and ROC-AUC

Compare individual models with an ensemble strategy

Dataset
Source: Kaggle – Asthma Disease Prediction Dataset

Size: 2,392 records

Features: 29 (demographics, clinical, environmental, lifestyle)

Target: Diagnosis (0 = No Asthma, 1 = Asthma)

Methodology
Preprocessing

Removed missing values and encoded categorical variables

Used one-hot encoding and label encoding

Applied SMOTE for class balance

Models

XGBoost (gradient-boosted decision trees)

SVM (RBF kernel with hyperparameter tuning)

ANN (2 hidden layers, ReLU, Adam optimizer)

Evaluation Metrics

F1-score

Recall (clinical sensitivity)

ROC-AUC

Confusion matrix

Results Summary

Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
XGBoost	0.9081	0.2000	0.0800	0.0833	0.6184
SVM	0.9415	0.2000	0.0400	0.0667	0.4555
ANN	0.8874	0.1500	0.2400	0.1481	0.7000
Key Findings
ANN showed the best balance between recall and F1-score, making it more suitable for clinical applications

XGBoost and SVM leaned toward predicting the majority class (non-asthma) despite SMOTE

ANN's ability to detect non-linear patterns improved asthma case detection

Tools and Libraries
Python 3

Jupyter Notebook

pandas, numpy, matplotlib, seaborn

scikit-learn, xgboost, imblearn

TensorFlow / Keras

Ethical Considerations
Dataset is anonymized and publicly licensed under Creative Commons CC0

Compliant with GDPR and University of Hertfordshire ethical guidelines

No human participants or personally identifiable information involved

How to Run
Clone the repository:
git clone https://github.com/AlekyaSaidam/Alekya-project.git

Install required packages:
pip install -r requirements.txt

Run the Jupyter notebook:
Open asthma_prediction.ipynb in Jupyter Notebook
