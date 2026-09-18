# ml_models/tests/test_preprocessing.py

import pandas as pd
import pytest
from ml_models.preprocessing import clean_text, validate_dataset, split_dataset

def test_clean_text():
    assert clean_text("Bonjour, tout le monde !") == "bonjour tout le monde"
    assert clean_text("L'Éléphant est gros...") == "léléphant est gros"
    assert clean_text(None) == ""

def test_validate_dataset():
    # Dataset valide avec 'content'
    df1 = pd.DataFrame([
        {"content": "Super produit !", "label": "1"},
        {"content": "Mauvais !", "label": "0"}
    ])
    validated1 = validate_dataset(df1)
    assert "text" in validated1.columns
    assert validated1.loc[0, "text"] == "super produit"
    assert validated1.loc[0, "label"] == 1
    
    # Dataset avec colonnes manquantes
    df2 = pd.DataFrame([{"text": "Seulement du texte"}])
    with pytest.raises(ValueError, match="Le dataset doit contenir une colonne"):
        validate_dataset(df2)

def test_split_dataset():
    df = pd.DataFrame([
        {"text": "a", "label": 1},
        {"text": "b", "label": 1},
        {"text": "c", "label": 0},
        {"text": "d", "label": 0}
    ])
    # Avec peu de données, split_dataset s'adapte sans crash
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.5)
    assert len(X_train) == 2
    assert len(X_test) == 2
