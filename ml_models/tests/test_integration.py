# ml_models/tests/test_integration.py

import os
import json
import pytest
from pathlib import Path
from jan_agent.database import SQLiteStore
from ml_models.trainer import train_and_evaluate
from ml_models.versioning import save_pipeline, load_pipeline, delete_pipeline_file
from ml_models.predictor import predict, predict_proba
from ml_models.evaluator import evaluate_pipeline

TEST_DB_PATH = "test_agent_memory.sqlite3"
TEST_DATASET_PATH = "test_dataset.json"

@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    # Setup: Créer un mini-dataset d'entraînement
    mini_dataset = [
        {"content": "C'est un excellent produit !", "label": "1"},
        {"content": "C'est absolument génial et rapide.", "label": "1"},
        {"content": "J'adore ce produit.", "label": "1"},
        {"content": "Très déçu, mauvaise qualité.", "label": "0"},
        {"content": "Le service client est horrible.", "label": "0"},
        {"content": "Ne fonctionne pas du tout, à fuir.", "label": "0"}
    ]
    with open(TEST_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(mini_dataset, f)
        
    yield
    
    # Teardown: Supprimer les fichiers temporaires
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    if os.path.exists(TEST_DATASET_PATH):
        os.remove(TEST_DATASET_PATH)
        
    # Nettoyer les fichiers de modèle générés par le test dans saved_models/test_model/
    test_models_dir = Path(__file__).parent.parent / "saved_models" / "test_model"
    if test_models_dir.exists():
        import shutil
        shutil.rmtree(test_models_dir)

def test_full_lifecycle():
    # 1. Initialiser le SQLiteStore de test
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    store = SQLiteStore(TEST_DB_PATH)
    
    # 2. Créer un modèle
    model_id = store.create_ml_model(
        name="test_model",
        description="Modèle de test",
        algorithm="LogisticRegression",
        hyperparams={"C": 1.0}
    )
    assert model_id is not None
    
    model = store.get_ml_model(model_id)
    assert model["name"] == "test_model"
    assert model["algorithm"] == "LogisticRegression"
    assert model["status"] == "draft"
    
    # 3. Entraîner le modèle
    train_res = train_and_evaluate(
        dataset_source=TEST_DATASET_PATH,
        algorithm_name=model["algorithm"],
        hyperparams=model["hyperparams"]
    )
    
    assert train_res["dataset_size"] == 6
    assert "pipeline" in train_res
    assert "metrics" in train_res
    assert train_res["metrics"]["accuracy"] > 0.0
    
    # 4. Versionner et sauvegarder le modèle
    version_tag = "v1.0.0"
    file_path = save_pipeline("test_model", version_tag, train_res["pipeline"])
    assert os.path.exists(file_path)
    
    version_id = store.create_ml_model_version(
        model_id=model_id,
        version_tag=version_tag,
        file_path=file_path,
        dataset_path=TEST_DATASET_PATH,
        dataset_size=train_res["dataset_size"],
        metrics=train_res["metrics"],
        training_duration_ms=train_res["training_duration_ms"],
        notes="Premier entraînement de test"
    )
    assert version_id is not None
    
    # Activer la version
    store.activate_ml_model_version(model_id, version_id)
    
    # Vérifier que le statut du modèle est mis à jour
    model = store.get_ml_model(model_id)
    assert model["status"] == "active"
    
    active_version = store.get_active_ml_model_version(model_id)
    assert active_version is not None
    assert active_version["version_tag"] == "v1.0.0"
    
    # 5. Prédictions
    pred_pos = predict(active_version["file_path"], "excellent produit et génial !")
    assert pred_pos == 1
    
    pred_neg = predict(active_version["file_path"], "horrible qualité et mauvais service.")
    assert pred_neg == 0
    
    probas = predict_proba(active_version["file_path"], "C'est génial")
    assert len(probas) == 2
    assert sum(probas.values()) == pytest.approx(1.0)
    
    # 6. Évaluation
    eval_res = evaluate_pipeline(active_version["file_path"], TEST_DATASET_PATH)
    assert eval_res["accuracy"] > 0.0
    
    # 7. Historique et nettoyage des fichiers
    store.add_ml_training_history(model_id, version_id, "train", {"tag": version_tag})
    history = store.get_ml_training_history(model_id)
    assert len(history) == 1
    assert history[0]["action"] == "train"
    
    # Nettoyer
    delete_pipeline_file(active_version["file_path"])
    assert not os.path.exists(file_path)
