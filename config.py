"""
Configuration file for Aspect-Based Sentiment Analysis (ABSA) system.
Defines file paths, aspect taxonomy, model hyperparameters, and UI settings.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset paths
DEFAULT_DATASET_PATH = RAW_DATA_DIR / "product_reviews.csv"
CLEANED_DATASET_PATH = PROCESSED_DATA_DIR / "cleaned_reviews.csv"
METRICS_JSON_PATH = REPORTS_DIR / "model_metrics.json"

# Saved Model Filepaths
TFIDF_VECTORIZER_PATH = SAVED_MODELS_DIR / "tfidf_vectorizer.joblib"
LOGISTIC_REGRESSION_PATH = SAVED_MODELS_DIR / "logistic_regression.joblib"
NAIVE_BAYES_PATH = SAVED_MODELS_DIR / "naive_bayes.joblib"
LABEL_ENCODER_PATH = SAVED_MODELS_DIR / "label_encoder.joblib"

# Supported Aspect Categories & Taxonomy Synonyms (Sorted by specificity)
ASPECT_CATEGORIES = {
    "Camera": [
        "camera quality", "picture quality", "photo quality", "video quality",
        "night mode", "camera", "photo", "photos", "picture", "pictures", 
        "image", "images", "video", "videos", "recording", "lens", 
        "sensor", "portrait", "selfie", "zoom", "shutter", "megapixels", "mp"
    ],
    "Battery": [
        "battery backup", "battery life", "charging speed", "fast charging", 
        "battery", "backup", "charging", "charger", "charge", 
        "mah", "drain", "power", "standby", "recharge", "endurance"
    ],
    "Display": [
        "display quality", "screen resolution", "refresh rate", "viewing angle",
        "touch response", "display", "screen", "panel", "oled", "amoled", 
        "lcd", "resolution", "brightness", "hz", "hdr", "touchscreen", "retina", "bezel"
    ],
    "Performance": [
        "processor speed", "gaming performance", "system speed",
        "performance", "speed", "processor", "cpu", "gpu", "ram", "chipset", 
        "fast", "slow", "lag", "laggy", "hanging", "multitasking", "gaming", "smooth", 
        "loading", "fps", "heating", "warm", "throttle", "snapdragon", "bionic"
    ],
    "Price": [
        "value for money", "price tag", "retail price", "pricing",
        "price", "cost", "value", "money", "budget", "expensive", "cheap", 
        "affordable", "overpriced", "worth", "deal", "discount"
    ],
    "Design": [
        "build quality", "exterior finish", "aesthetic design",
        "design", "look", "looks", "build", "finish", "feel", 
        "weight", "lightweight", "heavy", "sleek", "compact", "durability", 
        "material", "body", "aesthetic", "ergonomics"
    ],
    "Delivery": [
        "delivery service", "shipping speed", "product packaging", 
        "delivery", "shipping", "packaging", "package", "dispatch", "courier", 
        "arrival", "box", "transit", "delivered", "packed"
    ],
    "Customer Service": [
        "customer service", "customer support", "warranty assistance", "service center",
        "support team", "after sales", "warranty", "refund", "return policy", 
        "assistance", "staff", "guarantee"
    ],
    "Sound": [
        "sound quality", "audio quality", "speaker volume", "bass response",
        "sound", "audio", "speaker", "speakers", "volume", "bass", "clarity", 
        "mic", "microphone", "headphone jack", "earpiece", "treble", "noise cancellation"
    ],
    "Software": [
        "software experience", "operating system", "software updates", "user interface",
        "software", "os", "android", "ios", "ui", "interface", "update", 
        "updates", "apps", "bloatware", "bug", "bugs", "firmware"
    ]
}

# Inverted mapping: keyword -> canonical category (sorted with multi-word first)
KEYWORD_TO_ASPECT = {}
for category, keywords in ASPECT_CATEGORIES.items():
    for kw in keywords:
        KEYWORD_TO_ASPECT[kw.lower()] = category

# Sentiment labels & styling
SENTIMENT_COLORS = {
    "Positive": "#28a745",
    "Negative": "#dc3545",
    "Neutral": "#6c757d",
    "Mixed": "#fd7e14"
}

SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

# Model Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FEATURES = 6000
NGRAM_RANGE = (1, 2)
