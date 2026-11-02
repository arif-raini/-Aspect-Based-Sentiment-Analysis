"""
Unit and Integration Tests for Aspect-Based Sentiment Analysis (ABSA).
Tests text preprocessing, aspect extraction, model prediction, and end-to-end ABSA.
"""

import unittest
from preprocessing.text_preprocessing import TextPreprocessor
from models.aspect_extractor import AspectExtractor
from models.sentiment_model import HybridSentimentAnalyzer
from utils.dataset_loader import load_dataset


class TestABSAPipeline(unittest.TestCase):
    """Test suite for validating ABSA pipeline components."""

    @classmethod
    def setUpClass(cls):
        cls.preprocessor = TextPreprocessor()
        cls.aspect_extractor = AspectExtractor()
        cls.analyzer = HybridSentimentAnalyzer()

    def test_contraction_expansion(self):
        """Test expansion of common contractions."""
        text = "I don't like the phone, it doesn't work and isn't fast."
        expanded = self.preprocessor.expand_contractions(text)
        self.assertIn("do not", expanded)
        self.assertIn("does not", expanded)
        self.assertIn("is not", expanded)

    def test_negation_preservation(self):
        """Ensure negation words like 'not' are preserved for sentiment."""
        tokens = ["the", "camera", "is", "not", "good"]
        filtered = self.preprocessor.remove_stopwords(tokens)
        self.assertIn("not", filtered)

    def test_aspect_extraction_multi(self):
        """Test extraction of multiple distinct aspects from compound review."""
        review = "The phone camera is excellent, battery backup is poor, and the display quality is amazing."
        aspects = self.aspect_extractor.extract_aspect_contexts(review)
        aspect_names = {a["aspect"] for a in aspects}
        
        self.assertIn("Camera", aspect_names)
        self.assertIn("Battery", aspect_names)
        self.assertIn("Display", aspect_names)

    def test_clause_segmentation(self):
        """Verify complex reviews are cleanly partitioned into clauses."""
        text = "Screen is sharp, but the battery drains fast."
        clauses = self.aspect_extractor.segment_clauses(text)
        self.assertTrue(len(clauses) >= 2)

    def test_end_to_end_absa(self):
        """Verify full ABSA on the benchmark review sample."""
        review = "The phone camera is excellent, battery backup is poor, and the display quality is amazing."
        result = self.analyzer.analyze_review_absa(review, model_name="logistic_regression")
        
        aspect_map = {a["aspect"]: a["sentiment"] for a in result["aspects"]}
        
        self.assertEqual(aspect_map.get("Camera"), "Positive")
        self.assertEqual(aspect_map.get("Battery"), "Negative")
        self.assertEqual(aspect_map.get("Display"), "Positive")
        self.assertEqual(result["overall_sentiment"], "Mixed")

    def test_vader_baseline(self):
        """Test rule-based VADER predictor."""
        vader_res = self.analyzer.predict_vader("This product is absolutely amazing and perfect!")
        self.assertEqual(vader_res["sentiment"], "Positive")
        self.assertTrue(vader_res["polarity_scores"]["compound"] > 0.5)

    def test_dataset_loader(self):
        """Test dataset loading and integrity."""
        df = load_dataset()
        self.assertGreater(len(df), 500)
        self.assertIn("review_text", df.columns)
        self.assertIn("sentiment", df.columns)


if __name__ == "__main__":
    unittest.main()

