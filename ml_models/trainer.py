# ml_models/trainer.py

import time
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from ml_models.config import ALGORITHMS, DEFAULT_HYPERPARAMS, DEFAULT_TFIDF_PARAMS
from ml_models.preprocessing import load_dataset, validate_dataset, split_dataset

def build_pipeline(algorithm_name: str, hyperparams: dict = None, tfidf_params: dict = None) -> Pipeline:
    """Construit un pipeline sklearn TF-IDF + classifieur."""
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Algorithme '{algorithm_name}' non supporté. Supportés : {list(ALGORITHMS.keys())}")
    
    # Fusion des hyperparamètres avec ceux par défaut
    final_hyperparams = DEFAULT_HYPERPARAMS[algorithm_name].copy()
    if hyperparams:
        final_hyperparams.update(hyperparams)
        
    final_tfidf_params = DEFAULT_TFIDF_PARAMS.copy()
    if tfidf_params:
        final_tfidf_params.update(tfidf_params)
        
    # Conversion du format de ngram_range (list en tuple si nécessaire)
    if "ngram_range" in final_tfidf_params and isinstance(final_tfidf_params["ngram_range"], list):
        final_tfidf_params["ngram_range"] = tuple(final_tfidf_params["ngram_range"])
        
    # Instanciation de l'algorithme et du vectorizer
    vectorizer = TfidfVectorizer(**final_tfidf_params)
    classifier_cls = ALGORITHMS[algorithm_name]
    
    # Certains algorithmes ne prennent pas class_weight
    if "class_weight" in final_hyperparams and algorithm_name == "MultinomialNB":
        final_hyperparams.pop("class_weight")
        
    # Filtrer les hyperparamètres pour ne passer que ceux acceptés par la classe du modèle
    import inspect
    sig = inspect.signature(classifier_cls)
    valid_params = set(sig.parameters.keys())
    filtered_hyperparams = {k: v for k, v in final_hyperparams.items() if k in valid_params}
    
    classifier = classifier_cls(**filtered_hyperparams)
    
    return Pipeline([
        ("tfidf", vectorizer),
        ("model", classifier)
    ])

def train_and_evaluate(dataset_source: str, algorithm_name: str, hyperparams: dict = None, tfidf_params: dict = None) -> dict:
    """
    Charge le dataset, l'entraîne et l'évalue. Retourne le pipeline entraîné et les métriques associées.
    """
    # 1. Chargement et preprocessing
    raw_df = load_dataset(dataset_source)
    df = validate_dataset(raw_df)
    
    dataset_size = len(df)
    
    # 2. Split train/test
    X_train, X_test, y_train, y_test = split_dataset(df)
    
    # 3. Construction du pipeline
    pipeline = build_pipeline(algorithm_name, hyperparams, tfidf_params)
    
    # 4. Entraînement avec mesure du temps
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_duration_ms = (time.time() - start_time) * 1000
    
    # 5. Évaluation
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    
    # Génération du rapport de classification au format texte et dict
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classification_report": report_dict,
        "classification_report_text": report_text
    }
    
    return {
        "pipeline": pipeline,
        "dataset_size": dataset_size,
        "training_duration_ms": training_duration_ms,
        "metrics": metrics
    }
