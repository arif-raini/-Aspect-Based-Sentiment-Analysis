"""
Text Preprocessing Module for Aspect-Based Sentiment Analysis.
Performs text normalization, contraction expansion, tokenization,
negation-aware stop-word filtering, and POS-informed lemmatization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Comprehensive contraction mapping for sentiment preservation
CONTRACTIONS_DICT = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not"
}

# Negation and intensity words that must NOT be removed as stopwords
NEGATION_WORDS = {
    "not", "no", "nor", "neither", "never", "barely", "hardly", 
    "scarcely", "rarely", "seldom", "without", "against", "but", "however"
}

# Key sentiment participles that should remain as adjectives
SENTIMENT_PARTICIPLES = {
    "amazing", "outstanding", "pleasing", "charming", "stunning",
    "disappointing", "frustrating", "shocking", "annoying", "boring",
    "confusing", "overwhelming", "disgusting", "fascinating"
}


def get_wordnet_pos(word: str, nltk_tag: str) -> str:
    """Map NLTK POS tag to WordNet POS tag for accurate lemmatization."""
    if word.lower() in SENTIMENT_PARTICIPLES:
        return wordnet.ADJ
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


class TextPreprocessor:
    """
    Modular text preprocessor providing full pipeline and step-by-step inspection.
    """

    def __init__(self, preserve_negations: bool = True):
        self.lemmatizer = WordNetLemmatizer()
        self.preserve_negations = preserve_negations
        
        # Load NLTK stopwords and optionally remove sentiment-critical negation tokens
        base_stopwords = set(stopwords.words('english'))
        if self.preserve_negations:
            self.stop_words = base_stopwords - NEGATION_WORDS
        else:
            self.stop_words = base_stopwords

    def expand_contractions(self, text: str) -> str:
        """Expand English contractions into full words (e.g., don't -> do not)."""
        pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in CONTRACTIONS_DICT.keys()) + r')\b')
        expanded = pattern.sub(lambda x: CONTRACTIONS_DICT[x.group()], text.lower())
        
        expanded = re.sub(r"n\'t", " not", expanded)
        expanded = re.sub(r"\'re", " are", expanded)
        expanded = re.sub(r"\'s", " is", expanded)
        expanded = re.sub(r"\'d", " would", expanded)
        expanded = re.sub(r"\'ll", " will", expanded)
        expanded = re.sub(r"\'t", " not", expanded)
        expanded = re.sub(r"\'ve", " have", expanded)
        expanded = re.sub(r"\'m", " am", expanded)
        return expanded

    def clean_text(self, text: str) -> str:
        """
        Remove URLs, HTML tags, special symbols, and redundant whitespaces.
        Preserves alphanumerics and standard punctuation delimiters.
        """
        if not isinstance(text, str):
            return ""

        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        """Split text into words using NLTK word tokenizer."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Filter out stopwords while retaining negation tokens."""
        return [
            token for token in tokens
            if token.lower() not in self.stop_words and token not in string.punctuation
        ]

    def lemmatize_tokens(self, tokens: list[str]) -> list[str]:
        """Apply POS-tagged lemmatization using WordNet."""
        if not tokens:
            return []
        
        pos_tags = nltk.pos_tag(tokens)
        lemmatized = []
        for word, tag in pos_tags:
            wn_pos = get_wordnet_pos(word, tag)
            lemma = self.lemmatizer.lemmatize(word.lower(), pos=wn_pos)
            lemmatized.append(lemma)
        return lemmatized

    def preprocess(self, text: str) -> str:
        """
        Full end-to-end preprocessing pipeline returning a cleaned,
        lemmatized string suitable for TF-IDF vectorization.
        """
        if not text or not isinstance(text, str):
            return ""

        cleaned = self.clean_text(text)
        expanded = self.expand_contractions(cleaned)
        tokens = self.tokenize(expanded)
        no_stops = self.remove_stopwords(tokens)
        lemmas = self.lemmatize_tokens(no_stops)
        return " ".join(lemmas)

    def get_preprocessing_stages(self, text: str) -> dict:
        """
        Educational helper: returns each intermediate transformation step.
        Useful for UI display and university project demonstration.
        """
        if not text:
            return {}

        cleaned = self.clean_text(text)
        expanded = self.expand_contractions(cleaned)
        tokens = self.tokenize(expanded)
        no_punct = [t for t in tokens if t not in string.punctuation]
        no_stops = self.remove_stopwords(tokens)
        lemmas = self.lemmatize_tokens(no_stops)
        final_str = " ".join(lemmas)

        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "expanded_contractions": expanded,
            "raw_tokens": tokens,
            "tokens_no_punctuation": no_punct,
            "tokens_no_stopwords": no_stops,
            "lemmatized_tokens": lemmas,
            "final_processed_text": final_str
        }
