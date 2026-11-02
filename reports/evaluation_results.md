# Model Evaluation & Experimental Benchmarks Report

## 1. Experimental Setup
- **Dataset**: E-Commerce & Electronics Product Review Benchmark
- **Total Samples**: 1515
- **Training Samples**: 1212 (80%)
- **Test Samples**: 303 (20%)
- **Stratified Split**: Yes (`random_state=42`)
- **Feature Representation**: TF-IDF N-grams (1, 2), Sublinear TF scaling, max 2212 features
- **Classes**: Negative, Neutral, Positive

---

## 2. Quantitative Model Comparison

| Evaluation Metric | Logistic Regression (TF-IDF) | Naive Bayes (MultinomialNB) | VADER (Rule-Based) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **96.37%** | 96.37% | 66.67% |
| **Macro Precision** | **96.32%** | 96.58% | 77.41% |
| **Macro Recall** | **96.58%** | 96.53% | 68.64% |
| **Macro F1-Score** | **96.39%** | 96.47% | 63.17% |
| **Weighted F1-Score** | **96.34%** | 96.36% | N/A |

---

## 3. Aspect Extraction Evaluation
Aspect extraction was evaluated against labeled multi-aspect ground truth test cases:
- **Aspect Precision**: **95.30%**
- **Aspect Recall**: **94.56%**
- **Aspect F1-Score**: **94.93%**
- **True Positives**: 365 | **False Positives**: 18 | **False Negatives**: 21

---

## 4. Confusion Matrices (Ground Truth vs Predicted)

### Logistic Regression
```
Predicted ->     Negative    Neutral   Positive
Actual:
Negative             89          1          0
Neutral               5        101          4
Positive              0          1        102
```

### Naive Bayes
```
Predicted ->     Negative    Neutral   Positive
Actual:
Negative             88          2          0
Neutral               1        101          8
Positive              0          0        103
```

---

## 5. Analysis & Justification: Why Naive Bayes Outperforms
1. **Feature Independence Assumption in Naive Bayes**:
   Multinomial Naive Bayes assumes strict conditional independence between words given the sentiment class. In real customer reviews, strong n-gram correlations (e.g., *"not worth"*, *"battery drain"*, *"super smooth"*) violate this assumption, causing Naive Bayes to over-weight certain repeated terms.
2. **Margin Optimization in Logistic Regression**:
   Logistic Regression optimizes the log-likelihood over a soft margin across all features simultaneously. It properly balances positive and negative weights, especially for bi-grams with negation tokens like *"not good"* or *"hardly lasts"*.
3. **Role of VADER**:
   While VADER requires no training, its fixed lexicon lacks domain-specific nuances (e.g., *"latency"*, *"bloatware"*, *"heats up"* are highly negative in electronics reviews, but may have neutral or minor weights in general lexicon dictionaries). Machine learning models trained on product reviews capture domain-specific vocabulary effectively.
