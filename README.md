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

#task2

1. Audio Loading & Duration Control

We start by loading the audio files using Librosa, enforcing a fixed duration of 3.0 seconds for each sample.

Why?

Ensures consistency across all audio inputs

Prevents very long or very short clips from biasing the model

Makes batch processing and CNN input handling easier

2. Silence Removal (Preprocessing)

After loading, we apply silence removal to eliminate non-informative silent segments from the audio signal.


-Silence does not contribute to emotion recognition

-Reduces noise and irrelevant data

-Helps the model focus on meaningful speech patterns

3. Feature Extraction using MFCC

We extract MFCC (Mel-Frequency Cepstral Coefficients) from the processed audio signals.

Key points:

MFCC extraction implicitly applies framing and Hamming windowing

Framing splits the signal into short time frames

The Hamming window minimizes spectral leakage and smooths frame edges
MFCC:
-Closely matches human auditory perception
-Widely used and effective for speech and emotion recognition

4. Target Label Extraction

Target labels are extracted directly from the audio file names (e.g., emotion encoded in filename).

Features are stored as a NumPy array with object dtype : This is necessary because MFCC sequences may have variable time lengths before padding

5. Target Distribution Check

We visualize the target label distribution using plots: To check whether the dataset is balanced or imbalanced

Helps decide whether techniques like resampling or weighting are needed

6. Padding MFCC Sequences (Very Important)

We apply padding to the MFCC feature sequences to make them all the same shape.

why is required : 

CNNs and deep learning models require fixed-size inputs

Audio samples naturally produce MFCCs with different time steps

Padding ensures:

-Uniform tensor shape
-Batch processing compatibility
-Stable and efficient training
 Padding does not add new information; it only standardizes input size.

7. Input Reshaping for CNN

We add an extra dimension to MFCC features to make them compatible with Conv2D layers.

Conv2D expects 4D input : (samples, time, MFCC, channels)
MFCCs are treated like an image (time × frequency)

8. Train–Test Split & Standardization

The dataset is split into training and testing sets, followed by standardization.

Why standardization?

-Stabilizes gradients
-Improves convergence speed
-Prevents features with large values from dominating learning

9. Model Architecture: CNN + LSTM (Hybrid Model)
CNN Part:
-Extracts local spatial patterns from MFCCs
-Learns phonetic and spectral features
-Uses ReLU activation for non-linearity

(CNN treats MFCCs like a spectrogram image.)

LSTM Part : 

-Processes CNN output as a time sequence
-Captures temporal and frequency evolution over time
-Learns long-term dependencies in speech

LSTM ( how features change over time)

10. Output Layer - Compilation

Model is compiled with:

-Adam: Adaptive and efficient optimizer
-Sparse categorical crossentropy: Suitable when labels are integer-encoded (not one-hot)
-Accuracy: Simple and interpretable evaluation metric

Enhanced performance using ensemble boosting, with Logistic Regression as a meta-classifier to combine predictions and improve overall classification accuracy.
