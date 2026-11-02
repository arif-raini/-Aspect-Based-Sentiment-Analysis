# Aspect-Based Sentiment Analysis of Product Reviews

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20spaCy-green.svg)](https://spacy.io/)
[![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![UI](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)

A complete, production-grade Natural Language Processing (NLP) and Machine Learning mini-project that performs **Aspect-Based Sentiment Analysis (ABSA)** on customer product reviews. 

Instead of classifying an entire review with a single coarse sentiment, this system isolates individual product aspects (*Camera*, *Battery*, *Display*, *Performance*, *Price*, *Design*, *Delivery*, *Customer Service*, *Sound*, *Software*) and classifies their polarities independently.

---

## 🚀 Key Features

- **Live Single-Review ABSA**: Enter any review to instantly visualize preprocessed tokens, extracted aspect entities, isolated context clauses, aspect sentiments, and overall verdict.
- **Independent Clause-Level Classification**: Resolves compound multi-aspect sentences like:
  > *"The phone camera is excellent, battery backup is poor, and the display quality is amazing."*
  > - **Camera:** Positive
  > - **Battery:** Negative
  > - **Display:** Positive
  > - **Overall:** Mixed
- **Multi-Model Benchmark**: Compares **Logistic Regression (TF-IDF)**, **Multinomial Naive Bayes (TF-IDF)**, and **NLTK VADER (Rule-based Lexicon)**.
- **Batch CSV Analysis**: Upload customer review CSVs, run batch ABSA with progress tracking, filter results, and export analyzed data.
- **Interactive Visualizations**: Interactive Plotly and Seaborn charts for aspect distributions, confusion matrices, and model accuracy benchmarks.
- **University Viva Voce Guide & Report**: Includes 10-chapter academic documentation and 25+ Viva questions and answers.

---

## 📊 Experimental Results (80/20 Test Split)

| Evaluation Metric | Logistic Regression (TF-IDF) | Naive Bayes (MultinomialNB) | VADER (Rule-Based Baseline) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **96.37%** | 96.37% | 61.42% |
| **Macro Precision** | **96.42%** | 96.40% | 63.85% |
| **Macro Recall** | **96.35%** | 96.33% | 61.90% |
| **Macro F1-Score** | **96.37%** | 96.35% | 62.48% |
| **Aspect Extraction F1** | **94.93%** | **94.93%** | N/A |

---

## 🏗️ System Architecture

```
+-----------------------------------------------------------------------------------+
|                                Customer Product Review                            |
|  "The phone camera is excellent, battery backup is poor, and display is amazing." |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              NLP Preprocessing Engine                             |
|  - Lowercase Conversion & Contraction Expansion ("don't" -> "do not")             |
|  - Punctuation & Special Character Cleaning                                       |
|  - Negation-Aware Stop-word Removal (Preserves "not", "no", "never", "barely")    |
|  - POS-Informed Lemmatization (WordNet Lemmatizer)                                |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         Aspect Extraction & Segmentation                          |
|  - Clause Partitioning (Coordinating Conjunctions: but, however, and, yet)        |
|  - Dependency Parsing & Noun Chunks (spaCy: nsubj, dobj, amod, acomp)             |
|  - Aspect Taxonomy & Synonym Mapping (Longest matching compound priority)         |
|                                                                                   |
|    [Camera]  --> Context: "The phone camera is excellent"                         |
|    [Battery] --> Context: "battery backup is poor"                                |
|    [Display] --> Context: "and the display quality is amazing"                    |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       Sentiment Classification Engine                             |
|  - TF-IDF N-Gram Vectorizer (1, 2)                                                |
|  - Classifiers: Logistic Regression | Naive Bayes | VADER                         |
|                                                                                   |
|    Camera Context  --> [Positive] (Confidence: 85%)                               |
|    Battery Context --> [Negative] (Confidence: 89%)                               |
|    Display Context --> [Positive] (Confidence: 91%)                               |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           Overall Sentiment Synthesis                             |
|  - Multi-Polarity Resolution Logic:                                               |
|    Positive Aspects + Negative Aspects ==> Overall: [MIXED]                       |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       Interactive Streamlit Web Dashboard                         |
|  - Live Single Review ABSA | Batch CSV Processing | Analytics | Benchmarks        |
+-----------------------------------------------------------------------------------+
```

---

## 📁 Project Directory Structure

```
aspect-sentiment-analysis/
├── config.py                 # Paths, taxonomy dictionary, model hyperparameters
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation and quickstart guide
│
├── data/
│   ├── raw/                  # Raw benchmark product review CSV
│   └── processed/            # Cleaned, preprocessed cached data
│
├── preprocessing/
│   ├── __init__.py
│   └── text_preprocessing.py # Text normalization, lemmatization, contractions
│
├── models/
│   ├── __init__.py
│   ├── aspect_extractor.py   # Dependency parsing, clause segmentation, taxonomy
│   ├── sentiment_model.py    # Hybrid sentiment analyzer (LR, NB, VADER)
│   └── model_evaluation.py   # Benchmark evaluation and confusion matrix generator
│
├── utils/
│   ├── __init__.py
│   ├── dataset_loader.py     # CSV loading, validation, synthetic data synthesis
│   └── visualization.py      # Plotly interactive charts and heatmaps
│
├── notebooks/
│   └── experimentation.ipynb # Jupyter notebook for academic lab demonstration
│
├── reports/
│   ├── evaluation_results.md # Empirical metrics and confusion matrices
│   ├── academic_report.md    # 10-chapter university project report
│   └── viva_questions.md     # 25+ Viva Voce questions and answers
│
├── saved_models/             # Serialized joblib models and TF-IDF vectorizer
└── tests/
    └── test_pipeline.py      # Unit and integration tests
```

---

## ⚡ Installation & Setup Instructions

### 1. Prerequisites
- Python 3.11, 3.12, or 3.13 installed on your machine.
- Git (optional).

### 2. Clone or Navigate to the Project Directory
```bash
cd "c:\Users\raini\OneDrive\Desktop\NLP project"
```

### 3. Create and Activate Virtual Environment
On Windows:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 5. Download Required NLTK Corpora
```powershell
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng')"
```

---

## 🏃 Running the Application

Launch the Streamlit web dashboard:
```powershell
streamlit run app.py
```
The application will open automatically in your browser at:
`http://localhost:8501`

---

## 🧪 Running Unit Tests & Benchmarks

Run the complete test suite:
```powershell
python -m unittest tests/test_pipeline.py
```

Re-run model evaluation and regenerate benchmark metrics:
```powershell
python -c "from models.model_evaluation import evaluate_models; evaluate_models(force_retrain=True)"
```

---

## 🎓 Academic Documentation
- Complete 10-chapter Project Report: [`reports/academic_report.md`](reports/academic_report.md)
- Viva Voce Questions & Answers: [`reports/viva_questions.md`](reports/viva_questions.md)
- Empirical Model Metrics: [`reports/evaluation_results.md`](reports/evaluation_results.md)

