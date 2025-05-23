# Network Anomaly Detection System

## 📌 Project Overview
An anomaly detection system implementing:
- **Z-Score thresholding** for statistical outlier detection
- **Probability Density Function (PDF) analysis** for feature distribution
- **Naive Bayes classifiers** (GaussianNB, MultinomialNB, BernoulliNB) for attack prediction

Developed as part of NETW504 coursework at German University in Cairo.

## 📂 Repository Structure
```
├── Naive.Bayes_Estimation.py    # Naive Bayes implementation
├── PDF-Anomaly_detection.py     # Z-Score & PDF analysis
├── Train_data.csv               # Dataset
├── Report.docx                  # Project documentation
├── project description.docx     # Background & objectives
└── README.md                    # This file
```

## 🛠️ Installation
1. Clone repository:
```bash
git clone https://github.com/Mostafa-EISHinawi/[repository-name].git
```
2. Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## 🚀 Usage
Run anomaly detection:
```bash
python PDF-Anomaly_detection.py --input Train_data.csv
```

Run Naive Bayes classifier:
```bash
python Naive.Bayes_Estimation.py
```

## 📊 Key Features
- Automated threshold tuning for Z-Score detection
- PDF/PMF fitting for numerical/categorical features
- Performance metrics (Accuracy/Precision/Recall)
- Comparative analysis of Naive Bayes variants

## 📝 Documentation
- `Report.docx`: Detailed methodology and results
- `project description.docx`: Project background
---

*Last updated: 5/2024 
