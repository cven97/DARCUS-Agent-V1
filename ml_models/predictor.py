# ml_models/predictor.py

import logging
from typing import List, Dict, Any, Union
from ml_models.preprocessing import clean_text
from ml_models.versioning import load_pipeline

logger = logging.getLogger(__name__)

# Cache global des modèles chargés pour éviter les E/S disque répétées
_MODEL_CACHE = {}

def get_cached_pipeline(file_path: str):
    """Charge et met en cache le pipeline pour une prédiction ultra rapide."""
    if file_path not in _MODEL_CACHE:
        logger.info(f"Chargement du modèle dans le cache : {file_path}")
        _MODEL_CACHE[file_path] = load_pipeline(file_path)
    return _MODEL_CACHE[file_path]

def clear_model_cache(file_path: str = None):
    """Vide le cache de modèles, ou une entrée spécifique."""
    global _MODEL_CACHE
    if file_path:
        if file_path in _MODEL_CACHE:
            del _MODEL_CACHE[file_path]
    else:
        _MODEL_CACHE.clear()

def predict(file_path: str, text: str) -> Any:
    """Effectue une prédiction simple pour un seul texte."""
    pipeline = get_cached_pipeline(file_path)
    cleaned = clean_text(text)
    pred = pipeline.predict([cleaned])[0]
    
    # Conversion de types numpy pour la sérialisation JSON
    if hasattr(pred, "item"):
        return pred.item()
    return pred

def predict_batch(file_path: str, texts: List[str]) -> List[Any]:
    """Effectue des prédictions en lot pour une liste de textes."""
    pipeline = get_cached_pipeline(file_path)
    cleaned_texts = [clean_text(t) for t in texts]
    preds = pipeline.predict(cleaned_texts)
    
    return [p.item() if hasattr(p, "item") else p for p in preds]

def predict_proba(file_path: str, text: str) -> Dict[str, float]:
    """
    Retourne les probabilités pour chaque classe si le modèle le supporte.
    En cas de non support (comme LinearSVC), retourne les scores de décision.
    """
    pipeline = get_cached_pipeline(file_path)
    cleaned = clean_text(text)
    model = pipeline.named_steps["model"]
    
    classes = [str(c) for c in model.classes_]
    
    # Tentative d'utilisation de predict_proba
    if hasattr(pipeline, "predict_proba") or hasattr(model, "predict_proba"):
        probas = pipeline.predict_proba([cleaned])[0]
        return {classes[i]: float(probas[i]) for i in range(len(classes))}
        
    # Repli sur decision_function (ex: LinearSVC)
    elif hasattr(pipeline, "decision_function") or hasattr(model, "decision_function"):
        scores = pipeline.decision_function([cleaned])[0]
        # Si binaire
        if len(classes) == 2:
            # score unique, représentant la confiance pour la classe positive
            # sigmoid simplifiée pour le renvoi d'une "probabilité" indicative
            import math
            try:
                prob = 1 / (1 + math.exp(-scores))
            except OverflowError:
                prob = 0.0 if scores < 0 else 1.0
            return {classes[0]: 1.0 - prob, classes[1]: prob}
        else:
            return {classes[i]: float(scores[i]) for i in range(len(classes))}
            
    return {}
