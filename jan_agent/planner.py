import json
import logging
from .utils import call_llm

def generate_plan_steps(query, url, headers, model_id, timeout=60, planner_prompt=None, num_ctx=8192,
                         keep_alive=None, session=None):
    """
    Appelle le LLM pour générer un plan de vol (liste d'étapes) pour la requête donnée.
    Retourne une liste de strings (les étapes).
    """
    if not planner_prompt:
        planner_prompt = (
            "Tu es un planificateur expert. L'utilisateur a une requête. "
            "Ta tâche est de décomposer cette requête en un plan d'action séquentiel.\n"
            "RÈGLES IMPORTANTES :\n"
            "1. Ne réponds pas à la question de l'utilisateur.\n"
            "2. Fournis UNIQUEMENT un tableau JSON de chaînes de caractères.\n"
            "3. Le nombre d'étapes DOIT être proportionnel à la complexité. Pour une requête simple, génère EXACTEMENT 1 étape.\n"
            "4. N'ajoute AUCUN préfixe comme 'Étape 1 :' dans tes descriptions."
        )
    prompt = (
        f"{planner_prompt}\n\n"
        "Exemple pour une requête de listage de fichiers ('je veux voir mes fichiers') :\n"
        "[\n"
        "  \"Lister les fichiers du répertoire actuel\"\n"
        "]\n\n"
        "Exemple pour une question simple :\n"
        "[\n"
        "  \"Rechercher les informations clés et donner la réponse\"\n"
        "]\n\n"
        "Exemple pour une tâche complexe :\n"
        "[\n"
        "  \"Rechercher la documentation officielle du projet XYZ\",\n"
        "  \"Analyser les dépendances nécessaires\",\n"
        "  \"Rédiger le script de déploiement\"\n"
        "]\n\n"
        "Requête de l'utilisateur :\n"
        f"{query}"
    )

    messages = [
        {"role": "system", "content": "Tu es un assistant qui génère exclusivement du JSON valide."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = call_llm(
            server_url=url,
            headers=headers,
            model_id=model_id,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
            timeout=timeout,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
            session=session,
        )
        
        # Nettoyer la réponse pour extraire le JSON (au cas où le modèle ajoute des markdown)
        content = response.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        steps = json.loads(content)
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
            return steps
        else:
            return ["Traiter la requête en une seule étape (erreur de format)"]
            
    except Exception as e:
        logging.error(f"Erreur lors de la génération du plan : {e}")
        return ["Rechercher l'information principale", "Synthétiser la réponse"]
