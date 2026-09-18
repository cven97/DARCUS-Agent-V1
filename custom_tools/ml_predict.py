from jan_agent.tools import tool
from jan_agent.database import SQLiteStore
from ml_models.predictor import predict, predict_proba

@tool("ml_predict", "Prédit le label et retourne les probabilités de prédiction pour un texte donné à l'aide d'un modèle ML enregistré et actif.")
def ml_predict(model_name: str, text: str):
    """
    Analyse un texte avec un modèle ML entraîné et actif.
    Retourne la prédiction et les probabilités/confiance de chaque classe.
    """
    store = SQLiteStore("agent_memory.sqlite3")
    model = store.get_ml_model_by_name(model_name)
    if not model:
        return f"Erreur : Modèle '{model_name}' introuvable."
        
    active_version = store.get_active_ml_model_version(model["id"])
    if not active_version:
        return f"Erreur : Le modèle '{model_name}' n'a aucune version active. Veuillez d'abord l'entraîner et l'activer."
        
    try:
        pred = predict(active_version["file_path"], text)
        probas = predict_proba(active_version["file_path"], text)
        return {
            "model": model_name,
            "version": active_version["version_tag"],
            "prediction": pred,
            "probabilities": probas
        }
    except Exception as e:
        return f"Erreur lors de la prédiction : {str(e)}"
