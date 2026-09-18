# ml_models/config.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# Registre des algorithmes supportés et leurs constructeurs associés
ALGORITHMS = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "LinearSVC": LinearSVC,
    "MultinomialNB": MultinomialNB
}

# Hyperparamètres par défaut pour chaque algorithme
DEFAULT_HYPERPARAMS = {
    "LogisticRegression": {
        "class_weight": "balanced",
        "max_iter": 1000,
        "C": 1.0
    },
    "RandomForestClassifier": {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42
    },
    "LinearSVC": {
        "class_weight": "balanced",
        "max_iter": 1000,
        "C": 1.0
    },
    "MultinomialNB": {
        "alpha": 1.0
    }
}

# Configuration TF-IDF par défaut
DEFAULT_TFIDF_PARAMS = {
    "max_features": 2000,
    "ngram_range": [1, 2]  # Équivalent de (1, 2)
}
