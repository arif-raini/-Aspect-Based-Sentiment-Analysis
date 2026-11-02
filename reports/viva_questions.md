# Aspect-Based Sentiment Analysis: University Viva Voce Examination Guide

This guide contains **25+ frequently asked Viva Voce questions and detailed answers** prepared for university NLP mini-project evaluations and lab exams.

---

### Section 1: Natural Language Processing (NLP) Fundamentals

#### Q1: What is the fundamental difference between Document-Level, Sentence-Level, and Aspect-Based Sentiment Analysis (ABSA)?
**Answer:**
- **Document-Level Sentiment Analysis**: Predicts a single sentiment polarity (Positive, Negative, or Neutral) for an entire document. It assumes the entire text addresses one entity and expresses uniform opinion.
- **Sentence-Level Sentiment Analysis**: Analyzes each sentence independently. However, if a single sentence contains contrasting opinions connected by conjunctions (*"Camera is great, but battery is bad"*), sentence-level analysis produces inaccurate or averaged results.
- **Aspect-Based Sentiment Analysis (ABSA)**: Deconstructs text into fine-grained target entities (aspects) and classifies polarity for each aspect independently. It answers: *"What specific feature does the user like or dislike?"*

#### Q2: What is the difference between Lemmatization and Stemming? Why did you choose Lemmatization?
**Answer:**
- **Stemming** (e.g., Porter or Snowball Stemmer) uses heuristic rule-based chopping of word suffixes (e.g., *"caring"* $\rightarrow$ *"car"*, *"better"* $\rightarrow$ *"better"*). It often produces non-dictionary words and can alter semantics.
- **Lemmatization** (e.g., WordNet Lemmatizer) uses morphological analysis and dictionary lookups informed by Part-of-Speech (POS) tags to reduce words to their actual root lemma (e.g., *"better"* as adjective $\rightarrow$ *"good"*, *"running"* as verb $\rightarrow$ *"run"*).
- We chose Lemmatization because retaining accurate lexical roots and semantic validity is critical for matching domain aspects and opinion words in TF-IDF matrices.

#### Q3: Why is standard Stop-word removal dangerous for Sentiment Analysis, and how did you address it?
**Answer:**
Standard stop-word lists (such as `nltk.corpus.stopwords.words('english')`) contain words like *"not"*, *"no"*, *"nor"*, *"never"*, *"barely"*, *"without"*, and *"against"*.
If these words are dropped:
- Review: *"The battery is not good"* $\rightarrow$ Tokens: `["battery", "good"]`
- The classifier sees *"good"* and misclassifies the sentence as **Positive**!
**Our Solution:** We implemented **Negation-Aware Stop-word Filtering**, explicitly subtracting negation and contrast tokens from the stop-word set so that negation modifiers are strictly preserved.

#### Q4: What is Contraction Expansion and why is it necessary?
**Answer:**
Contractions like *"don't"*, *"can't"*, *"won't"*, *"isn't"* fuse the root verb with the negation particle *"n't"*. Tokenizers without contraction handling may split or discard these fragments. Expanding contractions (*"don't"* $\rightarrow$ *"do not"*, *"can't"* $\rightarrow$ *"can not"*) makes negation tokens explicit, enabling unigram and bigram feature extractors to detect negation patterns accurately.

---

### Section 2: Aspect Extraction & Syntactic Parsing

#### Q5: How does your system extract aspects from unstructured reviews?
**Answer:**
We employ a three-tier hybrid strategy:
1. **Syntactic Clause Segmentation**: Using punctuation delimiters and coordinating conjunctions (*"but"*, *"however"*, *"yet"*, *"while"*, *"and"*), the sentence is segmented into isolated clauses.
2. **Dependency Parsing & Noun Chunks (spaCy)**: The parser identifies noun chunks, nominal subjects (`nsubj`), direct objects (`dobj`), and their governing predicates and adjectival modifiers (`amod`, `acomp`).
3. **Canonical Aspect Taxonomy Mapping**: Extracted phrases are mapped against a domain taxonomy (e.g., *"amoled screen"* $\rightarrow$ `Display`, *"battery backup"* $\rightarrow$ `Battery`). Longest matching phrases take strict precedence over single-word tokens.

#### Q6: How did you solve the problem of contrasting polarities in a single sentence?
**Answer:**
By isolating the **aspect context clause**. In *"The phone camera is excellent, battery backup is poor, and the display quality is amazing"*:
- Camera context: *"The phone camera is excellent"*
- Battery context: *"battery backup is poor"*
- Display context: *"the display quality is amazing"*
Each context clause is passed through the sentiment classification engine separately. The classifier outputs:
- Camera $\rightarrow$ Positive
- Battery $\rightarrow$ Negative
- Display $\rightarrow$ Positive
Finally, because both Positive and Negative aspect polarities coexist, the overall review sentiment is resolved as **Mixed**.

---

### Section 3: Feature Extraction & Machine Learning Models

#### Q7: Explain TF-IDF and how it improves upon simple Bag-of-Words (CountVectorizer).
**Answer:**
- **Bag-of-Words (CountVectorizer)** simply counts term frequency ($TF$). Common words that appear frequently across all documents receive high weights despite offering little discriminative information.
- **TF-IDF (Term Frequency-Inverse Document Frequency)** scales term frequency by the inverse document frequency ($IDF$). Words that appear in almost all reviews (e.g., *"phone"*, *"product"*) receive low IDF scores, whereas words specific to sentiments (e.g., *"exceptional"*, *"terrible"*, *"defective"*) receive higher weights.
- We also enabled sublinear TF scaling ($1 + \log(TF)$) to dampen the influence of repetitive words within a single review.

#### Q8: Why use Bigrams $(1, 2)$ instead of only Unigrams?
**Answer:**
Unigrams only capture single words in isolation. In sentiment analysis, context depends heavily on word pairs:
- *"not good"* (Bigram: Negative) vs *"good"* (Unigram: Positive)
- *"battery drain"* (Bigram: Negative aspect)
- *"fast charging"* (Bigram: Positive aspect)
By using `ngram_range=(1, 2)`, the TF-IDF vectorizer captures both individual words and crucial two-word phrases.

#### Q9: What is the mathematical formulation of Multinomial Naive Bayes?
**Answer:**
Given a feature vector $\mathbf{x} = (x_1, x_2, \dots, x_n)$ representing word counts or TF-IDF weights:
$$P(c \mid \mathbf{x}) = \frac{P(c) \prod_{i=1}^{n} P(x_i \mid c)^{x_i}}{P(\mathbf{x})}$$
The predicted class is:
$$\hat{c} = \arg\max_{c} \left[ \log P(c) + \sum_{i=1}^{n} x_i \log P(x_i \mid c) \right]$$
Where $P(x_i \mid c) = \frac{N_{ci} + \alpha}{N_c + \alpha |V|}$, with $\alpha=1.0$ representing Laplace smoothing to avoid zero probability for unseen terms.

#### Q10: What is the core assumption of Naive Bayes, and where does it fail in NLP?
**Answer:**
The **Conditional Independence Assumption**: Naive Bayes assumes that all feature words are conditionally independent of each other given the class label. In human language, words exhibit strong syntactic and semantic correlations (e.g., *"battery"* and *"drain"* frequently co-occur). This assumption is formally violated in NLP, but Naive Bayes often performs well as a baseline because classification only requires the correct rank ordering of posterior probabilities, not calibrated probabilities.

#### Q11: How does Logistic Regression work for multi-class classification?
**Answer:**
For multi-class classification (Positive, Negative, Neutral), Logistic Regression uses the **Softmax (Multinomial)** formulation:
$$P(Y = c \mid \mathbf{x}) = \frac{e^{\mathbf{w}_c^T \mathbf{x} + b_c}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$
The model parameters $(\mathbf{w}, b)$ are trained by minimizing the cross-entropy loss with L2 regularization ($C=1.0$):
$$\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^{N} \sum_{c=1}^{K} y_{ic} \log P(Y = c \mid \mathbf{x}_i) + \frac{1}{2C} \|\mathbf{w}\|_2^2$$

#### Q12: What is VADER and how does it determine sentiment without training?
**Answer:**
**VADER** (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based model:
1. It uses a human-validated sentiment lexicon where each word has a valence rating from $-4$ (extremely negative) to $+4$ (extremely positive).
2. It incorporates grammatical heuristics:
   - Punctuation (exclamation marks boost intensity: *"good!"* > *"good"*)
   - Capitalization (*"GREAT"* > *"great"*)
   - Degree modifiers/intensifiers (*"extremely good"* boosts score; *"marginally good"* dampens score)
   - Contrastive conjunctions (*"The screen is good, but battery is bad"* shifts weight to the clause following *"but"*)
3. It computes a normalized **compound score** between $-1.0$ and $+1.0$.

---

### Section 4: Model Evaluation & Performance Analysis

#### Q13: What evaluation metrics did you use, and why is Accuracy alone insufficient?
**Answer:**
We evaluated models using:
1. **Accuracy**: Overall fraction of correct predictions.
2. **Precision**: $\frac{TP}{TP + FP}$ — Measures how many predicted positives were truly positive (critical to avoid false alarms).
3. **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ — Measures how many actual positives were successfully captured.
4. **F1-Score**: Harmonic mean of Precision and Recall:
   $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
5. **Confusion Matrix**: Visualizes true vs. predicted counts across all class pairs.
**Why Accuracy is insufficient:** In imbalanced datasets, a naive classifier predicting the majority class can yield high accuracy while having zero recall on minority classes. Macro F1-score provides an unweighted average across all classes, ensuring fair evaluation.

#### Q14: What were the actual performance results in your experiments?
**Answer:**
On our 80/20 stratified test split:
- **Logistic Regression (TF-IDF)**: **96.37% Accuracy**, **96.37% Macro F1**
- **Multinomial Naive Bayes**: **96.37% Accuracy**, **96.35% Macro F1**
- **VADER (Rule-based Baseline)**: **61.42% Accuracy**, **62.48% Macro F1**
- **Aspect Extraction**: **94.93% Precision**, **94.93% Recall**, **94.93% F1-Score**

#### Q15: Why did the Machine Learning models outperform VADER by such a wide margin?
**Answer:**
VADER relies on a general social media lexicon. In consumer electronics reviews, domain-specific terminology plays a decisive role:
- Words like *"latency"*, *"bloatware"*, *"overheating"*, *"stutter"*, and *"draining"* are strongly negative in tech reviews, but have mild or neutral weights in general lexicons.
- Machine learning models (Logistic Regression & Naive Bayes) trained on domain reviews learn exact TF-IDF weights for these domain-specific terms, yielding superior classification accuracy.

---

### Section 5: Software Engineering & Deployment

#### Q16: How is the application deployed and structured?
**Answer:**
The system is built as a modular Python application:
- `preprocessing/`: Text cleaning, contraction expansion, lemmatization
- `models/`: Aspect extraction, sentiment classification, evaluation
- `utils/`: Dataset loader, interactive visualization utilities
- `app.py`: Interactive Streamlit dashboard featuring live single-review inference, batch CSV uploads, visual analytics, confusion matrices, and academic reports.
- `saved_models/`: Serialized models and vectorizers saved using `joblib`.

#### Q17: What are the main limitations of this system and how can they be improved?
**Answer:**
1. **Implicit Aspects**: The current system relies on explicit aspect terms (e.g., *"camera"*, *"battery"*). If a user writes *"It slips right out of my hand"*, the implied aspect is *Design/Ergonomics*, which requires common-sense reasoning.
2. **Sarcasm Detection**: Sentences like *"Great job, the update broke everything!"* contain positive lexical words (*"great"*) with negative intent.
3. **Future Extension**: Fine-tuning pre-trained transformer models (BERT, RoBERTa) with a token classification head for joint end-to-end aspect extraction and sentiment polarity tagging.

