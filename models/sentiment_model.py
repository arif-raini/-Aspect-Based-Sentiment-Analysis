"""
Sentiment Analysis Engine supporting Logistic Regression, Naive Bayes, and VADER.
Provides aspect-level sentiment classification, probability estimation,
and overall review sentiment synthesis.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from config import (
    TFIDF_VECTORIZER_PATH,
    LOGISTIC_REGRESSION_PATH,
    NAIVE_BAYES_PATH,
    LABEL_ENCODER_PATH,
    MAX_FEATURES,
    NGRAM_RANGE,
    RANDOM_STATE
)
from preprocessing.text_preprocessing import TextPreprocessor
from models.aspect_extractor import AspectExtractor


class HybridSentimentAnalyzer:
    """
    Unified Sentiment Engine for Aspect-Based Sentiment Analysis (ABSA).
    Integrates TF-IDF + Logistic Regression, TF-IDF + Naive Bayes,
    and Rule-based VADER Sentiment Intensity Analyzer.
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.aspect_extractor = AspectExtractor()
        self.vader = SentimentIntensityAnalyzer()
        
        self.vectorizer: TfidfVectorizer | None = None
        self.log_reg: LogisticRegression | None = None
        self.naive_bayes: MultinomialNB | None = None
        self.label_encoder: LabelEncoder | None = None
        
        self._load_saved_models_if_present()

    def _load_saved_models_if_present(self) -> bool:
        """Attempt to load pre-trained models from disk."""
        if (
            TFIDF_VECTORIZER_PATH.exists()
            and LOGISTIC_REGRESSION_PATH.exists()
            and NAIVE_BAYES_PATH.exists()
            and LABEL_ENCODER_PATH.exists()
        ):
            try:
                self.vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
                self.log_reg = joblib.load(LOGISTIC_REGRESSION_PATH)
                self.naive_bayes = joblib.load(NAIVE_BAYES_PATH)
                self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
                return True
            except Exception:
                return False
        return False

    def is_trained(self) -> bool:
        """Check if ML models are initialized and trained."""
        return (
            self.vectorizer is not None
            and self.log_reg is not None
            and self.naive_bayes is not None
            and self.label_encoder is not None
        )

    def train_models(self, texts: list[str], labels: list[str]) -> dict:
        """
        Train both Logistic Regression and Multinomial Naive Bayes models
        using TF-IDF vectorization.
        """
        # Preprocess texts
        processed_texts = [self.preprocessor.preprocess(t) for t in texts]

        # Fit TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            sublinear_tf=True
        )
        X = self.vectorizer.fit_transform(processed_texts)

        # Label encoding
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)

        # Train Logistic Regression
        self.log_reg = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=1.0,
            random_state=RANDOM_STATE
        )
        self.log_reg.fit(X, y)

        # Train Naive Bayes
        self.naive_bayes = MultinomialNB(alpha=1.0)
        self.naive_bayes.fit(X, y)

        # Save artifacts
        self.save_models()

        return {
            "num_samples": len(texts),
            "vocab_size": len(self.vectorizer.vocabulary_),
            "classes": list(self.label_encoder.classes_)
        }

    def save_models(self) -> None:
        """Persist trained estimators and vectorizer to disk."""
        TFIDF_VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, TFIDF_VECTORIZER_PATH)
        joblib.dump(self.log_reg, LOGISTIC_REGRESSION_PATH)
        joblib.dump(self.naive_bayes, NAIVE_BAYES_PATH)
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)

    def predict_vader(self, text: str) -> dict:
        """
        Classify sentiment using NLTK VADER lexicon-based heuristics.
        Thresholds: compound >= 0.05 -> Positive, <= -0.05 -> Negative, else Neutral.
        """
        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]
        
        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Normalized confidence metric derived from compound magnitude
        confidence = abs(compound)
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "polarity_scores": scores
        }

    def predict_single_text(self, text: str, model_name: str = "logistic_regression") -> dict:
        """
        Classifies the sentiment of a single piece of text (or aspect context clause)
        using the specified model.
        
        model_name options: 'logistic_regression', 'naive_bayes', 'vader'
        """
        if model_name.lower() == "vader":
            return self.predict_vader(text)

        if not self.is_trained():
            # Fallback to VADER if ML models haven't been trained yet
            return self.predict_vader(text)

        processed = self.preprocessor.preprocess(text)
        if not processed:
            # If all tokens were stripped or text is empty, fall back to VADER
            return self.predict_vader(text)

        X = self.vectorizer.transform([processed])
        
        if model_name.lower() == "naive_bayes":
            model = self.naive_bayes
        else:
            model = self.log_reg

        # Model probability output
        probs = model.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        
        # Note: clearly document this is model estimated class probability, not absolute certainty
        est_prob = float(probs[pred_idx])

        # Also compute VADER compound polarity for reference
        vader_scores = self.vader.polarity_scores(text)

        class_probabilities = {
            self.label_encoder.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(probs)
        }

        return {
            "sentiment": pred_label,
            "confidence": round(est_prob, 3),
            "class_probabilities": class_probabilities,
            "vader_polarity": vader_scores["compound"]
        }

    def analyze_review_absa(self, review_text: str, model_name: str = "logistic_regression") -> dict:
        """
        Main Aspect-Based Sentiment Analysis execution pipeline:
        1. Preprocesses the text and extracts processing stages.
        2. Discovers aspects and isolates aspect context clauses.
        3. Classifies sentiment for each aspect independently.
        4. Synthesizes overall sentiment (handles Mixed polarities).
        """
        if not review_text or not isinstance(review_text, str):
            return {
                "original_text": "",
                "processed_text": "",
                "aspects": [],
                "overall_sentiment": "Neutral",
                "overall_score": 0.0
            }

        # Step 1: Preprocessing inspection
        stages = self.preprocessor.get_preprocessing_stages(review_text)
        processed_text = stages.get("final_processed_text", "")

        # Step 2: Aspect Context Extraction
        aspect_entries = self.aspect_extractor.extract_aspect_contexts(review_text)

        aspect_results = []
        sentiments_found = []

        # Step 3: Classify each aspect independently using its specific context
        for item in aspect_entries:
            aspect_name = item["aspect"]
            context_text = item["context"]
            
            # Predict sentiment on localized aspect context
            prediction = self.predict_single_text(context_text, model_name=model_name)
            
            aspect_results.append({
                "aspect": aspect_name,
                "raw_term": item["raw_term"],
                "context_clause": context_text,
                "sentiment": prediction["sentiment"],
                "confidence": prediction.get("confidence", 0.0),
                "details": prediction
            })
            sentiments_found.append(prediction["sentiment"])

        # Step 4: Overall review sentiment determination
        if not aspect_results:
            # Fallback if no specific aspects detected: classify whole review
            whole_pred = self.predict_single_text(review_text, model_name=model_name)
            overall_sentiment = whole_pred["sentiment"]
            overall_score = whole_pred.get("confidence", 0.5)
        else:
            has_positive = "Positive" in sentiments_found
            has_negative = "Negative" in sentiments_found
            has_neutral = "Neutral" in sentiments_found

            if has_positive and has_negative:
                overall_sentiment = "Mixed"
            elif has_positive and not has_negative:
                overall_sentiment = "Positive"
            elif has_negative and not has_positive:
                overall_sentiment = "Negative"
            else:
                overall_sentiment = "Neutral"
            
            # Compute average confidence across aspects
            confidences = [a["confidence"] for a in aspect_results if "confidence" in a]
            overall_score = round(sum(confidences) / len(confidences), 3) if confidences else 0.5

        return {
            "original_text": review_text,
            "processed_text": processed_text,
            "stages": stages,
            "aspects": aspect_results,
            "overall_sentiment": overall_sentiment,
            "overall_score": overall_score,
            "model_used": model_name
        }

