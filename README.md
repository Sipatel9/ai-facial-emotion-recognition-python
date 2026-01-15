# AI Facial Emotion Recognition using Python

## Overview
This project implements a facial emotion recognition (FER) system using classical machine learning techniques in Python.

The system detects and classifies facial emotions such as happiness, sadness, anger, fear, surprise, and disgust using image-based features.

## Technologies Used
- Python
- OpenCV
- NumPy
- Scikit-learn
- HOG (Histogram of Oriented Gradients)
- LBP (Local Binary Patterns)
- Support Vector Machine (SVM)
- Decision Tree Classifier

## Dataset
- JAFFE Dataset
- CK+ Dataset

## Methodology
- Face images were converted to grayscale and resized
- Feature extraction using HOG and LBP
- Classification using SVM (JAFFE) and Decision Tree (CK+)
- Performance evaluated using accuracy and confusion matrices

## Results
- SVM + HOG (JAFFE): 49.0% accuracy
- Decision Tree + LBP (CK+): 37.76% accuracy

## Notes
This project focuses on interpretable and lightweight machine learning models suitable for small datasets and low-resource environments.
