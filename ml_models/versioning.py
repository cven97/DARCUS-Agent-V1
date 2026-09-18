# ml_models/versioning.py

import os
import shutil
from pathlib import Path
import joblib

SAVED_MODELS_DIR = Path(__file__).parent / "saved_models"

def get_model_dir(model_name: str) -> Path:
    """Retourne le répertoire dédié à un modèle spécifique."""
    model_dir = SAVED_MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def save_pipeline(model_name: str, version_tag: str, pipeline) -> str:
    """Sauvegarde le pipeline complet dans un fichier .pkl."""
    model_dir = get_model_dir(model_name)
    file_path = model_dir / f"{version_tag}.pkl"
    joblib.dump(pipeline, file_path)
    return str(file_path)

def load_pipeline(file_path: str):
    """Charge le pipeline depuis le chemin spécifié."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier de modèle '{file_path}' n'existe pas.")
    return joblib.load(file_path)

def delete_pipeline_file(file_path: str) -> bool:
    """Supprime le fichier de modèle de la mémoire disque."""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            # Supprime le dossier parent s'il est vide
            parent_dir = os.path.dirname(file_path)
            if parent_dir != str(SAVED_MODELS_DIR) and not os.listdir(parent_dir):
                shutil.rmtree(parent_dir)
            return True
        except Exception:
            return False
    return False
