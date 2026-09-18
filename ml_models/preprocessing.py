# ml_models/preprocessing.py

import os
import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text: str) -> str:
    """Nettoie le texte en le convertissant en minuscules et en retirant la ponctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def load_dataset(source: str) -> pd.DataFrame:
    """
    Charge un dataset depuis diverses sources :
    - Un chemin de fichier JSON (ex: 'dataset.json')
    - Un chemin de fichier CSV (ex: 'dataset.csv')
    - Un nom de projet pr_generator sous le format 'pr_generator:<nom_projet>' (ex: 'pr_generator:sentiment_project')
    """
    if source.startswith("pr_generator:"):
        project_name = source.split(":", 1)[1]
        return _load_from_pr_generator(project_name)
    
    if not os.path.exists(source):
        raise FileNotFoundError(f"Le fichier dataset '{source}' n'existe pas.")

    if source.endswith(".json"):
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif source.endswith(".csv"):
        df = pd.read_csv(source)
    else:
        raise ValueError("Format de fichier non supporté. Utilisez .json ou .csv")

    return df

def _load_from_pr_generator(project_name: str) -> pd.DataFrame:
    """Charge les données générées par le module pr_generator."""
    from pr_generator.router import factory
    session = factory.Session()
    try:
        project = session.query(factory.Project).filter_by(name=project_name).first()
        if not project:
            raise ValueError(f"Le projet pr_generator '{project_name}' n'existe pas.")
        
        rows = session.query(factory.GeneratedData).filter_by(project_id=project.id).all()
        
        data = []
        for r in rows:
            data.append({
                "content": r.content,
                "label": r.label
            })
        
        if not data:
            raise ValueError(f"Aucune donnée trouvée pour le projet pr_generator '{project_name}'.")
        
        return pd.DataFrame(data)
    finally:
        session.close()

def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valide et normalise le dataset :
    - S'assure de la présence des colonnes nécessaires (content/text et label)
    - Renomme 'content' en 'text' si nécessaire
    - Convertit la colonne 'label' en entier ou string propre
    - Nettoie les textes
    """
    if df.empty:
        raise ValueError("Le dataset est vide.")

    # Normalisation du nom de la colonne de texte
    if "content" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"content": "text"})
    
    if "text" not in df.columns:
        raise ValueError("Le dataset doit contenir une colonne 'text' ou 'content'.")
    if "label" not in df.columns:
        raise ValueError("Le dataset doit contenir une colonne 'label'.")

    # Nettoyage et suppression des valeurs nulles
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).apply(clean_text)
    
    # Nettoyage des labels (suppression d'espaces et normalisation)
    df["label"] = df["label"].astype(str).str.strip()
    
    # Si tous les labels peuvent être convertis en entiers, on le fait
    try:
        # On teste si tous sont des entiers
        df["label"] = df["label"].astype(float).astype(int)
    except ValueError:
        # Sinon on garde en string (ex: classification multi-classe textuelle)
        pass

    return df

def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Sépare le dataset en ensembles de train et de test avec stratification."""
    # S'il n'y a qu'une seule classe ou pas assez d'échantillons pour stratifier
    label_counts = df["label"].value_counts()
    if len(label_counts) < 2 or label_counts.min() < 2:
        # Split simple sans stratification
        return train_test_split(
            df["text"],
            df["label"],
            test_size=test_size,
            random_state=random_state
        )
    
    return train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]
    )
