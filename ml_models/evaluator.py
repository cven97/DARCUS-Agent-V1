# ml_models/evaluator.py

import time
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from ml_models.preprocessing import load_dataset, validate_dataset
from ml_models.predictor import get_cached_pipeline

def evaluate_pipeline(file_path: str, dataset_source: str) -> dict:
    """Évalue un pipeline sur un dataset de test donné."""
    # 1. Chargement et validation
    raw_df = load_dataset(dataset_source)
    df = validate_dataset(raw_df)
    
    pipeline = get_cached_pipeline(file_path)
    
    # 2. Prédictions
    y_true = df["label"]
    y_pred = pipeline.predict(df["text"])
    
    # 3. Calcul des métriques
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, zero_division=0)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dataset_size": len(df),
        "classification_report": report_dict,
        "classification_report_text": report_text
    }

def compare_pipelines(file_paths: dict, dataset_source: str) -> dict:
    """
    Compare plusieurs pipelines côte à côte sur le même dataset de test.
    file_paths: dict {nom_version: chemin_pkl}
    """
    comparison_results = {}
    for name, path in file_paths.items():
        try:
            metrics = evaluate_pipeline(path, dataset_source)
            comparison_results[name] = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "dataset_size": metrics["dataset_size"]
            }
        except Exception as e:
            comparison_results[name] = {"error": str(e)}
            
    return comparison_results

def benchmark_pipeline(file_path: str, num_samples: int = 100) -> dict:
    """Mesure la vitesse d'inférence (benchmarking) sur un ensemble de phrases factices."""
    pipeline = get_cached_pipeline(file_path)
    
    test_sentences = [
        "C'est un excellent produit que je recommande vivement à tout le monde.",
        "Le service client est déplorable et le produit ne fonctionne pas du tout.",
        "Livraison un peu lente mais conforme à mes attentes.",
        "Rien à dire, tout est parfait et le vendeur est sérieux.",
        "Déçu par la qualité générale, je demande un remboursement immédiat."
    ] * (num_samples // 5 + 1)
    test_sentences = test_sentences[:num_samples]
    
    # Préchauffage
    pipeline.predict([test_sentences[0]])
    
    start_time = time.time()
    for sentence in test_sentences:
        pipeline.predict([sentence])
    duration = time.time() - start_time
    
    avg_latency_ms = (duration / num_samples) * 1000
    throughput = num_samples / duration
    
    return {
        "num_samples": num_samples,
        "total_duration_seconds": duration,
        "average_latency_ms": avg_latency_ms,
        "throughput_predictions_per_second": throughput
    }
