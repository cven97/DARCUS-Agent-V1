# ml_models/router.py

import os
import json
import threading
import jwt
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from jan_agent.database import SQLiteStore
from ml_models.config import ALGORITHMS
from ml_models.trainer import train_and_evaluate
from ml_models.versioning import save_pipeline, delete_pipeline_file
from ml_models.predictor import predict, predict_batch, predict_proba
from ml_models.evaluator import evaluate_pipeline, compare_pipelines, benchmark_pipeline

router = APIRouter(prefix="/api/models", tags=["ML Model Lifecycle"])
store = SQLiteStore("agent_memory.sqlite3")

# ─────────────────────────────────────────────
# Sécurité JWT (Miroir de jan_agent.api.main)
# ─────────────────────────────────────────────
JWT_SECRET = os.environ.get("JWT_SECRET", "change-me-in-production-use-a-long-random-string-of-32+chars")
JWT_ALGORITHM = "HS256"
security = HTTPBearer()

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("sub", "")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token a expiré")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token invalide")

# ─────────────────────────────────────────────
# Modèles de données Pydantic
# ─────────────────────────────────────────────
class ModelCreateRequest(BaseModel):
    name: str = Field(..., example="sentiment_fr")
    description: Optional[str] = Field("", example="Modèle de classification de sentiments en français")
    algorithm: str = Field("LogisticRegression", example="LogisticRegression")
    hyperparams: Optional[Dict[str, Any]] = Field(default_factory=dict, example={"C": 1.0, "max_iter": 1000})

class ModelUpdateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    hyperparams: Optional[Dict[str, Any]] = Field(default_factory=dict)

class TrainRequest(BaseModel):
    dataset_path: str = Field(..., example="dataset.json ou pr_generator:sentiment_project")
    version_tag: str = Field(..., example="v1.0.0")
    notes: Optional[str] = Field(None, example="Premier entraînement avec régression logistique")

class PredictRequest(BaseModel):
    text: str = Field(..., example="C'est un produit absolument fantastique !")

class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., example=["Super !", "Très mauvais service."])

class EvaluateRequest(BaseModel):
    dataset_path: str = Field(..., example="dataset.json")

# ─────────────────────────────────────────────
# Logique de Background Training
# ─────────────────────────────────────────────
def bg_train_model(task_id: int, model_id: int, dataset_path: str, algorithm: str, hyperparams: dict, version_tag: str, notes: str = None):
    try:
        # Séparer les hyperparamètres TF-IDF si présents
        tfidf_params = None
        clean_hyperparams = hyperparams.copy() if hyperparams else {}
        if clean_hyperparams and "tfidf" in clean_hyperparams:
            tfidf_params = clean_hyperparams.pop("tfidf")
            
        # Exécuter l'entraînement
        train_res = train_and_evaluate(
            dataset_source=dataset_path,
            algorithm_name=algorithm,
            hyperparams=clean_hyperparams,
            tfidf_params=tfidf_params
        )
        
        pipeline = train_res["pipeline"]
        dataset_size = train_res["dataset_size"]
        duration_ms = train_res["training_duration_ms"]
        metrics = train_res["metrics"]
        
        # Récupérer les détails actuels du modèle
        model = store.get_ml_model(model_id)
        if not model:
            raise ValueError(f"Le modèle ID {model_id} a été supprimé pendant l'entraînement.")
            
        model_name = model["name"]
        
        # Sauvegarder le fichier .pkl du pipeline
        file_path = save_pipeline(model_name, version_tag, pipeline)
        
        # Enregistrer la version en base de données
        version_id = store.create_ml_model_version(
            model_id=model_id,
            version_tag=version_tag,
            file_path=file_path,
            dataset_path=dataset_path,
            dataset_size=dataset_size,
            metrics=metrics,
            training_duration_ms=duration_ms,
            notes=notes
        )
        
        # Auto-activation s'il n'y a pas d'autres versions
        existing_versions = store.get_ml_model_versions(model_id)
        if len(existing_versions) <= 1:
            store.activate_ml_model_version(model_id, version_id)
            status = "active"
        else:
            status = "ready"
            
        store.update_ml_model(
            model_id=model_id,
            name=model["name"],
            description=model["description"],
            hyperparams=model["hyperparams"],
            status=status
        )
        
        # Historique
        history_details = {
            "version_tag": version_tag,
            "dataset_path": dataset_path,
            "metrics": {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"]
            },
            "duration_ms": duration_ms
        }
        store.add_ml_training_history(model_id, version_id, "train", history_details)
        
        # Terminer la tâche SQLite avec succès
        store.finish_task(task_id, result=json.dumps({"version_id": version_id, "version_tag": version_tag, "metrics": metrics}, ensure_ascii=False))
        
    except Exception as e:
        import traceback
        error_msg = f"Erreur d'entraînement : {str(e)}\n{traceback.format_exc()}"
        store.finish_task(task_id, error=error_msg)
        
        # Reset le statut du modèle en cas d'erreur
        model = store.get_ml_model(model_id)
        if model:
            versions = store.get_ml_model_versions(model_id)
            has_active = any(v["is_active"] == 1 for v in versions)
            status = "active" if has_active else "draft"
            store.update_ml_model(
                model_id=model_id,
                name=model["name"],
                description=model["description"],
                hyperparams=model["hyperparams"],
                status=status
            )
            
        store.add_ml_training_history(model_id, None, "train_failed", {"error": str(e)})

# ─────────────────────────────────────────────
# Endpoints FastAPI
# ─────────────────────────────────────────────

@router.get("/algorithms", dependencies=[Depends(verify_jwt)])
def list_algorithms():
    """Lister tous les algorithmes disponibles."""
    return {"algorithms": list(ALGORITHMS.keys())}

@router.post("", status_code=201, dependencies=[Depends(verify_jwt)])
def create_model(req: ModelCreateRequest):
    """Créer un nouveau modèle."""
    if req.algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Algorithme invalide. Choisissez parmi : {list(ALGORITHMS.keys())}")
    
    # Vérifier l'unicité du nom
    existing = store.get_ml_model_by_name(req.name)
    if existing:
        raise HTTPException(status_code=400, detail=f"Un modèle portant le nom '{req.name}' existe déjà.")
        
    model_id = store.create_ml_model(
        name=req.name,
        description=req.description,
        algorithm=req.algorithm,
        hyperparams=req.hyperparams
    )
    return {"id": model_id, "message": "Modèle créé avec succès"}

@router.get("", dependencies=[Depends(verify_jwt)])
def list_models():
    """Lister tous les modèles enregistrés."""
    return store.get_ml_models()

@router.get("/{model_id}", dependencies=[Depends(verify_jwt)])
def get_model_details(model_id: int):
    """Obtenir les détails d'un modèle avec ses versions."""
    model = store.get_ml_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
    
    versions = store.get_ml_model_versions(model_id)
    model["versions"] = versions
    return model

@router.put("/{model_id}", dependencies=[Depends(verify_jwt)])
def update_model(model_id: int, req: ModelUpdateRequest):
    """Modifier la configuration générale d'un modèle."""
    model = store.get_ml_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
        
    # Vérifier l'unicité du nom s'il change
    if req.name != model["name"]:
        existing = store.get_ml_model_by_name(req.name)
        if existing:
            raise HTTPException(status_code=400, detail=f"Un modèle portant le nom '{req.name}' existe déjà.")

    store.update_ml_model(
        model_id=model_id,
        name=req.name,
        description=req.description,
        hyperparams=req.hyperparams,
        status=model["status"]
    )
    return {"message": "Modèle mis à jour"}

@router.delete("/{model_id}", dependencies=[Depends(verify_jwt)])
def delete_model(model_id: int):
    """Supprimer un modèle, toutes ses versions et ses fichiers associés."""
    model = store.get_ml_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
        
    # Supprimer les fichiers physiques associés aux versions du modèle
    versions = store.get_ml_model_versions(model_id)
    for version in versions:
        delete_pipeline_file(version["file_path"])
        
    store.delete_ml_model(model_id)
    return {"message": "Modèle et toutes ses versions supprimés"}

@router.post("/{model_id}/train", status_code=202, dependencies=[Depends(verify_jwt)])
def train_model_endpoint(model_id: int, req: TrainRequest):
    """
    Lance l'entraînement du modèle sur un dataset spécifié.
    L'exécution s'effectue en arrière-plan via un thread et crée une tâche de progression.
    """
    model = store.get_ml_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
        
    if model["status"] == "training":
        raise HTTPException(status_code=400, detail="Ce modèle est déjà en cours d'entraînement.")
        
    # Vérifier que le tag de version n'est pas déjà pris
    versions = store.get_ml_model_versions(model_id)
    if any(v["version_tag"] == req.version_tag for v in versions):
        raise HTTPException(status_code=400, detail=f"La version '{req.version_tag}' existe déjà pour ce modèle.")
        
    # Mettre à jour le statut du modèle immédiatement
    store.update_ml_model(
        model_id=model_id,
        name=model["name"],
        description=model["description"],
        hyperparams=model["hyperparams"],
        status="training"
    )
    
    # Créer une tâche système SQLite pour le suivi
    task_id = store.create_task(
        kind="ml_train",
        payload={"model_id": model_id, "version_tag": req.version_tag, "dataset_path": req.dataset_path},
        status="running"
    )
    
    # Lancement du background thread
    thread = threading.Thread(
        target=bg_train_model,
        args=(task_id, model_id, req.dataset_path, model["algorithm"], model["hyperparams"], req.version_tag, req.notes),
        daemon=True
    )
    thread.start()
    
    return {"task_id": task_id, "message": "Entraînement démarré en arrière-plan"}

@router.get("/{model_id}/versions", dependencies=[Depends(verify_jwt)])
def list_model_versions(model_id: int):
    """Lister toutes les versions/checkpoints d'un modèle."""
    model = store.get_ml_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
    return store.get_ml_model_versions(model_id)

@router.post("/{model_id}/versions/{version_id}/activate", dependencies=[Depends(verify_jwt)])
def activate_version(model_id: int, version_id: int):
    """Activer une version spécifique (déploiement en production)."""
    model = store.get_ml_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
        
    version = store.get_ml_model_version(version_id)
    if not version or version["model_id"] != model_id:
        raise HTTPException(status_code=404, detail="Version introuvable pour ce modèle")
        
    store.activate_ml_model_version(model_id, version_id)
    store.add_ml_training_history(model_id, version_id, "deploy", {"version_tag": version["version_tag"]})
    return {"message": f"Version '{version['version_tag']}' activée avec succès"}

@router.delete("/{model_id}/versions/{version_id}", dependencies=[Depends(verify_jwt)])
def delete_version(model_id: int, version_id: int):
    """Supprimer une version de modèle."""
    model = store.get_ml_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
        
    version = store.get_ml_model_version(version_id)
    if not version or version["model_id"] != model_id:
        raise HTTPException(status_code=404, detail="Version introuvable pour ce modèle")
        
    if version["is_active"] == 1:
        raise HTTPException(status_code=400, detail="Impossible de supprimer la version active. Veuillez en activer une autre d'abord.")
        
    # Supprimer le fichier .pkl
    delete_pipeline_file(version["file_path"])
    store.delete_ml_model_version(version_id)
    return {"message": "Version supprimée"}

@router.post("/{model_id}/evaluate", dependencies=[Depends(verify_jwt)])
def evaluate_model_endpoint(model_id: int, req: EvaluateRequest):
    """Évaluer la version active du modèle sur un jeu de test."""
    model = store.get_ml_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
        
    active_version = store.get_active_ml_model_version(model_id)
    if not active_version:
        raise HTTPException(status_code=400, detail="Aucune version active disponible pour évaluation.")
        
    try:
        metrics = evaluate_pipeline(active_version["file_path"], req.dataset_path)
        # Historique
        store.add_ml_training_history(
            model_id=model_id,
            version_id=active_version["id"],
            action="evaluate",
            details={"dataset_path": req.dataset_path, "metrics": {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"]
            }}
        )
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_id}/predict")
def predict_endpoint(model_id: int, req: PredictRequest):
    """Prédire le label pour un texte avec la version active du modèle."""
    model = store.get_ml_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
        
    active_version = store.get_active_ml_model_version(model_id)
    if not active_version:
        raise HTTPException(status_code=400, detail="Aucune version active disponible pour la prédiction.")
        
    try:
        pred_label = predict(active_version["file_path"], req.text)
        probas = predict_proba(active_version["file_path"], req.text)
        return {"prediction": pred_label, "probabilities": probas}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_id}/predict/batch")
def predict_batch_endpoint(model_id: int, req: PredictBatchRequest):
    """Prédictions en lot avec la version active du modèle."""
    model = store.get_ml_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
        
    active_version = store.get_active_ml_model_version(model_id)
    if not active_version:
        raise HTTPException(status_code=400, detail="Aucune version active disponible pour la prédiction.")
        
    try:
        preds = predict_batch(active_version["file_path"], req.texts)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}/history", dependencies=[Depends(verify_jwt)])
def get_model_history(model_id: int):
    """Historique complet de toutes les actions sur un modèle."""
    model = store.get_ml_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
    return store.get_ml_training_history(model_id)

@router.get("/{model_id}/benchmark", dependencies=[Depends(verify_jwt)])
def benchmark_model(model_id: int, samples: int = 100):
    """Lancer un benchmark de vitesse d'inférence de la version active."""
    model = store.get_ml_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
        
    active_version = store.get_active_ml_model_version(model_id)
    if not active_version:
        raise HTTPException(status_code=400, detail="Aucune version active disponible pour le benchmark.")
        
    try:
        return benchmark_pipeline(active_version["file_path"], samples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}/compare", dependencies=[Depends(verify_jwt)])
def compare_versions(model_id: int, dataset_path: str):
    """Comparer toutes les versions de ce modèle sur un dataset commun."""
    model = store.get_ml_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
        
    versions = store.get_ml_model_versions(model_id)
    if not versions:
        raise HTTPException(status_code=400, detail="Aucune version disponible pour comparaison.")
        
    file_paths = {v["version_tag"]: v["file_path"] for v in versions}
    try:
        return compare_pipelines(file_paths, dataset_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
