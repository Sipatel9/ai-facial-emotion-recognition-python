# AI Facial Emotion Recognition (Classical ML) 🤖

Python-based AI facial emotion recognition system using classical machine learning algorithms (HOG, LBP, SVM) on JAFFE and CK+ datasets.

## 📋 Overview

This project implements emotion recognition using:
- **Classical ML Algorithms**: HOG (Histogram of Oriented Gradients), LBP (Local Binary Patterns)
- **Feature Extraction**: Multiple feature representation methods
- **Classification**: Support Vector Machines and Decision Trees
- **Datasets**: JAFFE & CK+ emotion datasets
- **High Interpretability**: Explainable machine learning approach
- **Lightweight**: Suitable for resource-constrained environments

## 🎯 Features

✅ Multiple feature extraction methods (HOG, LBP)  
✅ SVM and Decision Tree classification  
✅ Cross-validation for robust evaluation  
✅ Confusion matrix analysis  
✅ Real-time emotion prediction  
✅ Detailed performance metrics  
✅ Lightweight and efficient  
✅ No GPU required  

## 🧬 Emotions Recognized

- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😐 Neutral
- 😲 Surprised
- 😨 Fear
- 🤮 Disgust

## 📊 Dataset Information

| Dataset | Images | Emotions | Resolution | Format |
|---------|--------|----------|------------|--------|
| JAFFE | 213 | 7 | 256×256 | Grayscale |
| CK+ | 593 | 8 | 640×490 | Color |

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.8
scikit-learn >= 1.0
scikit-image >= 0.18
numpy >= 1.20
pandas >= 1.3
opencv-python >= 4.5
matplotlib >= 3.4
seaborn >= 0.11
```

### Installation
```bash
git clone https://github.com/Sipatel9/ai-facial-emotion-recognition-python.git
cd ai-facial-emotion-recognition-python
pip install -r requirements.txt
```

### Usage
```python
from emotion_recognizer import EmotionClassifier

# Initialize classifier
clf = EmotionClassifier(model='hog_svm')

# Train on dataset
clf.train('data/jaffe', 'data/ck_plus')

# Predict emotion from image
emotion, confidence = clf.predict('test_image.jpg')
print(f"Detected emotion: {emotion} (confidence: {confidence:.2%})")
```

## 📚 Feature Extraction Methods

### Histogram of Oriented Gradients (HOG)
- **Principle**: Captures edge and corner information
- **Robustness**: Handles lighting variations
- **Configuration**: 9×9 cell blocks with 9 orientation bins
- **Output**: 1,764-dimensional feature vector
- **Performance**: Good for structured patterns

### Local Binary Patterns (LBP)
- **Principle**: Captures texture information
- **Efficiency**: Computationally lightweight
- **Configuration**: 8-neighbor circular sampling
- **Output**: Histogram of local patterns
- **Performance**: Excellent for micro-expressions

### Combination (HOG + LBP)
- Concatenates both features for richer representation
- Balances edge and texture information
- Improved accuracy on diverse expressions

## 📈 Classification Performance

### JAFFE Dataset
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM + HOG | 49.0% | 0.52 | 0.49 | 0.48 |
| SVM + LBP | 42.0% | 0.45 | 0.42 | 0.41 |
| SVM + HOG+LBP | 51.0% | 0.55 | 0.51 | 0.50 |

### CK+ Dataset
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Decision Tree + LBP | 37.76% | 0.40 | 0.38 | 0.37 |
| SVM + HOG | 45.0% | 0.48 | 0.45 | 0.44 |
| SVM + HOG+LBP | 52.0% | 0.56 | 0.52 | 0.51 |

## 🏗️ Model Architecture

```
Image Input (256×256 grayscale)
    ↓
Preprocessing
├── Histogram Equalization
└── Normalization
    ↓
Feature Extraction
├── HOG Features
└── LBP Features
    ↓
Feature Vector Concatenation
    ↓
SVM/Decision Tree Classifier
    ↓
Emotion Prediction (7-8 classes)
```

## 🎓 Key Techniques

- **Grid Search**: Optimal hyperparameter tuning (C, gamma, kernel)
- **Cross-Validation**: K-fold validation (k=5) for robust evaluation
- **Feature Scaling**: StandardScaler normalization
- **Class Weighting**: Handle class imbalance in datasets
- **ROC-AUC**: Multi-class performance evaluation
- **Confusion Matrices**: Detailed error analysis

## 📁 Project Structure

```
├── data/
│   ├── jaffe/
│   │   ├── angry/
│   │   ├── happy/
│   │   └── ...
│   └── ck_plus/
│       ├── angry/
│       ├── happy/
│       └── ...
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── feature_extraction.ipynb
│   ├── model_training.ipynb
│   └── evaluation_analysis.ipynb
├── src/
│   ├── feature_extraction.py
│   ├── emotion_recognizer.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── models/
│   ├── emotion_svm_hog.pkl
│   ├── emotion_svm_lbp.pkl
│   └── emotion_svm_combined.pkl
└── README.md
```

## 🔬 Experimental Workflow

### 1. Data Preparation
```python
# Load and preprocess images
images, labels = load_dataset('data/jaffe')
images = preprocess(images)
```

### 2. Feature Extraction
```python
# Extract HOG and LBP features
hog_features = extract_hog(images)
lbp_features = extract_lbp(images)
combined_features = np.concatenate([hog_features, lbp_features], axis=1)
```

### 3. Model Training
```python
# Train SVM with grid search
clf = GridSearchCV(SVC(), param_grid, cv=5)
clf.fit(features, labels)
```

### 4. Evaluation
```python
# Evaluate performance
predictions = clf.predict(test_features)
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy:.2%}")
```

## 💡 Advantages of Classical ML

✅ **Minimal Computational Requirements** - No GPU needed  
✅ **Fast Training** - Minutes instead of hours  
✅ **Interpretable** - Feature importance visible  
✅ **Small Training Data** - Works with limited samples  
✅ **Low Latency** - Real-time inference  
✅ **Easy Deployment** - Lightweight models  
✅ **No Framework Dependency** - Pure Python  

## 🤝 Contributing

Contributions welcome! Help us:
- Improve feature extraction methods
- Test additional algorithms
- Optimize model performance
- Add new emotion classes
- Expand dataset support

## 🔗 Resources

- 📘 [Scikit-learn Documentation](https://scikit-learn.org/)
- 📘 [Scikit-image Documentation](https://scikit-image.org/)
- 📊 [JAFFE Dataset](https://www.kasrl.org/jaffe.html)
- 📊 [CK+ Dataset](https://www.jeffcohn.com/databases/)

## 📝 License

MIT License - Open for research and education

## 👩‍💻 Author

**Samira Patel**  
BSc (Hons) Computer Science  
University of Central Lancashire (UCLan)

## 📞 Contact

Questions about the implementation? [Reach out](https://github.com/Sipatel9)

---

**⭐ If you find this useful, please star the repository!**
