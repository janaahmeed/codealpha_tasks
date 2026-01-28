# codealpha_tasks
This repository contains Machine Learning projects completed as part of the CodeAlpha ML Internship. It includes real-world applications such as credit scoring, speech emotion recognition, and handwritten character recognition using Python, Scikit-learn, TensorFlow, and deep learning techniques with proper evaluation metrics.

in detail
#Task1 
1. Data Exploration

Loaded the dataset and examined its shape, data types, and general information.

Checked for missing (null) values to understand data quality.

Identified duplicate rows to avoid bias in model training.

2. Data Preprocessing

Applied Label Encoding to categorical (object-type) features so they can be used correctly in mathematical computations and machine learning models.

Inspected feature distributions to decide whether scaling was required.

Checked for values with NaN meaning and handled them using imputation techniques.

3. Feature Analysis & Selection

Generated a correlation heatmap to analyze relationships between features.

Removed or ignored features with low correlation to the target variable, based on a defined threshold, to reduce noise and improve model efficiency.

4. Handling Class Imbalance

Analyzed the target class distribution.

Applied SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance and improve the model’s ability to learn minority-class patterns.

5. Feature Scaling

Applied RobustScaler to selected continuous numerical features.

This scaler is especially suitable for financial and credit datasets, which often contain outliers and skewed distributions, as it relies on the median and interquartile range (IQR) instead of the mean and standard deviation.

6. Model Building
Used a RandomForestClassifier as the base learner.

Enhanced performance using ensemble boosting, with Logistic Regression as a meta-classifier to combine predictions and improve overall classification accuracy.
Used a RandomForestClassifier as the base learner.

Enhanced performance using ensemble boosting, with Logistic Regression as a meta-classifier to combine predictions and improve overall classification accuracy.
