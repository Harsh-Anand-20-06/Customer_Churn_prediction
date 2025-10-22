#  Customer Churn Prediction using Neural Networks (Keras)

This project focuses on predicting **Customer Churn** — i.e., whether a customer will discontinue a service — using a **Neural Network built with Keras**.  
It demonstrates complete data preprocessing, model training, evaluation, and **class imbalance handling** techniques such as **undersampling, oversampling, and SMOTE**.

---

##  Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset Description](#dataset-description)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Model Training](#model-training)
7. [Model Evaluation](#model-evaluation)
8. [Handling Class Imbalance](#handling-class-imbalance)
9. [Results and Discussion](#results-and-discussion)
10. [Key Insights](#key-insights)
11. [Technologies Used](#technologies-used)
12. [How to Run](#how-to-run)
13. [Author](#author)

---

##  Project Overview

Customer churn prediction helps companies **identify customers likely to stop using their service** so that targeted retention strategies can be implemented.  

In this project, a **binary classification model** was developed using **Keras** to predict churn (1) or not (0).  
Various techniques were used to improve the model’s robustness against **imbalanced data**, a common challenge in real-world datasets.

---

##  Problem Statement

> Predict whether a customer will churn (leave the service) based on their demographic and service-related information.

**Type:** Binary Classification  
**Target Variable:** `Churn` (0 = No, 1 = Yes)

---

##  Dataset Description

The dataset contains customer information such as:
- Demographics (gender, senior citizen, partner, dependents)
- Service information (InternetService, Contract type, Payment method)
- Account details (tenure, monthly charges, total charges)

**Target column:** `Churn`  
**Imbalance Observation:** Significantly more non-churn (0) samples than churn (1).

---

##  Data Preprocessing

### 1️ Encoding Categorical Data
- Applied **One-Hot Encoding** for features such as `InternetService` (DSL, Fiber Optic, No).
- Converted all categorical columns into **numeric format** suitable for neural networks.

### 2️ Scaling
- Applied **feature scaling** (e.g., StandardScaler or MinMaxScaler) to normalize continuous features.

### 3️ Splitting Data
- Formed feature matrix `X` and target vector `y`.
- Split into **training and testing sets** for model evaluation.

---

##  Model Architecture

A **Feedforward Neural Network (Multi-Layer Perceptron)** was implemented using Keras.

| Layer | Type | Activation | Purpose |
|-------|------|-------------|----------|
| Input | Dense | ReLU | Capture non-linear relationships |
| Hidden 1 | Dense | ReLU | Deep feature extraction |
| Hidden 2 | Dense | ReLU | Further learning of patterns |
| Output | Dense (1 unit) | Sigmoid | Binary classification (0 or 1) |

**Optimizer:** `Adam`  
**Loss Function:** `Binary Crossentropy`  
**Metric:** `Accuracy`

---

##  Model Training

- Trained the model on the processed dataset.
- Used **train-test split** to evaluate generalization performance.
- Checked **precision, recall, F1-score, and accuracy** for both classes.

---

##  Model Evaluation (Before Handling Imbalance)

| Metric | Class 0 | Class 1 |
|---------|----------|----------|
| Precision | 0.82 | 0.68 |
| Recall | 0.90 | 0.51 |
| F1-score | 0.86 | 0.59 |

**Accuracy:** 0.79  
**Observation:**  
- High accuracy but **poor recall and F1-score for Class 1** (churned customers).  
- Indicates **class imbalance** — the model favors majority (non-churn) class.

---

##  Handling Class Imbalance

Since churn (1) was the minority class, various rebalancing techniques were explored.

---

###  1. Undersampling

**Idea:** Reduce majority class samples to match the minority count.

| Metric | Class 0 | Class 1 |
|---------|----------|----------|
| Precision | 0.76 | 0.78 |
| Recall | 0.79 | 0.75 |
| F1-score | 0.77 | 0.77 |
| **Accuracy** | **0.77** |

 **Balanced and consistent performance** across both classes.  
 F1-score for churn improved significantly from **0.59 → 0.77**.

---

###  2. Oversampling

**Idea:** Duplicate minority class samples to balance both classes.

| Metric | Class 0 | Class 1 |
|---------|----------|----------|
| Precision | 0.79 | 0.74 |
| Recall | 0.72 | 0.80 |
| F1-score | 0.75 | 0.77 |
| **Accuracy** | **0.76** |

 Improved minority (churn) performance.  
 Slight drop in majority precision but more balanced F1-scores.

---

###  3. SMOTE (Synthetic Minority Oversampling Technique)

**Idea:** Generate synthetic examples of minority class using interpolation.

| Metric | Class 0 | Class 1 |
|---------|----------|----------|
| Precision | 0.88 | 0.54 |
| Recall | 0.79 | 0.69 |
| F1-score | 0.83 | 0.61 |
| **Accuracy** | **0.76** |

 SMOTE improved recall for churn class but **precision dropped**, resulting in a moderate F1.

---

##  Results and Discussion

| Technique | Accuracy | F1 (Class 1) | Observation |
|------------|-----------|--------------|--------------|
| Original (Imbalanced) | 0.79 | 0.59 | Poor minority detection |
| Undersampling | 0.77 | 0.77 | Balanced results |
| Oversampling | 0.76 | 0.77 | Consistent F1 improvement |
| SMOTE | 0.76 | 0.61 | Moderate gain in recall |

 **Undersampling and Oversampling** yielded the best balance between classes.  
 Accuracy alone can be misleading for imbalanced data — **F1-score** is the key metric.

---

##  Key Insights

- Handling data imbalance is crucial for fair model performance.  
- **Undersampling** is effective when dataset size is large enough.  
- **Oversampling/SMOTE** can be used when data loss is undesirable.  
- F1-score is more informative than accuracy for churn problems.  
- Preprocessing (encoding + scaling) strongly influences neural network stability.

---

##  Technologies Used

| Category | Tools / Libraries |
|-----------|-------------------|
| Programming Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Resampling Techniques | imbalanced-learn (for SMOTE) |

---


---

##  How to Run the Project

1. **Clone this repository**
   ```bash
   git clone https://github.com/Harsh-Anand-20-06/Customer_Churn_prediction.git
   cd Customer_Churn_prediction
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Open Jupyter Notebook**
   ```bash
   jupyter notebook
4. **Run the notebook**
  - Open customer_churn_prediction.ipynb
  - Execute cells sequentially.


##  Author

**Harsh Anand**  
B.Tech in Mechanical Engineering  
Indian Institute of Technology, Indore

- Email: sdeharsh2005@gmail.com  
- GitHub: https://github.com/Harsh-Anand-20-06
