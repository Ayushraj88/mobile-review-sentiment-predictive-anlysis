📊 Predictive Analytics on Mobile Reviews using Python & Machine Learning

This repository contains an end-to-end predictive analytics project built using Python and Machine Learning techniques. The project focuses on analyzing a real-world mobile reviews dataset to extract insights and build predictive models using both regression and classification approaches.

📁 Project Overview

The goal of this project is to:

Understand customer behavior through Exploratory Data Analysis (EDA)

Predict mobile ratings using regression models

Classify mobile price categories (Low / Mid / High) using classification algorithms

Compare model performance using standard evaluation metrics

🔧 Workflow

Data Loading

Loaded mobile reviews dataset using Pandas

Data Cleaning & Preprocessing

Handled missing values using median and mode imputation

Encoded categorical variables using Label Encoding

Standardized numerical features using StandardScaler

Exploratory Data Analysis (EDA)

Rating distribution analysis

Brand-wise average rating comparison

Correlation heatmap for numerical features

Relationship analysis between rating, price, and review characteristics

Feature Engineering

Review length

Word count

Price (USD)

User engagement indicators

🤖 Machine Learning Models
🔹 Regression Models

Linear Regression

Polynomial Regression (degree = 2)

Evaluation Metrics:

RMSE (Root Mean Squared Error)

R² Score

🔹 Classification Models

Decision Tree Classifier

K-Nearest Neighbors (K = 5)

Gaussian Naive Bayes

Target Variable:

Price segmented into Low / Mid / High using quantile-based binning

Evaluation Metrics:

Accuracy Score

Confusion Matrix

Classification Report

📈 Model Evaluation & Selection

All models were evaluated and compared using appropriate metrics.
The best-performing classification model was selected based on accuracy and confusion matrix analysis.

🛠️ Tech Stack

Language: Python

Libraries:

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn
