from jan_agent.tools import tool
from pr_generator.router import factory
import json
from sqlalchemy import text

@tool("get_generator_projects", "Liste tous les projets disponibles dans le générateur de données avec leurs variables d'entrée associées.")
def get_generator_projects():
    """
    Récupère la liste de tous les projets de génération de données de pr_generator.
    Retourne la liste des projets sous forme de chaîne JSON.
    """
    session = factory.Session()
    try:
        projects = session.query(factory.Project).all()
        result = []
        for p in projects:
            vars_list = json.loads(p.variables) if p.variables else ["text"]
            result.append({
                "id": p.id,
                "name": p.name,
                "variables": vars_list
            })
        return json.dumps(result, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        return f"Erreur lors de la récupération des projets : {str(e)}"
    finally:
        session.close()

@tool("get_generator_project_schema", "Retourne la structure/schéma (variables d'entrée) d'un projet du générateur de données spécifié.")
def get_generator_project_schema(project_name: str):
    """
    Récupère le schéma (les variables configurées) d'un projet pr_generator spécifique sous forme de chaîne JSON.
    """
    session = factory.Session()
    try:
        project = session.query(factory.Project).filter_by(name=project_name).first()
        if not project:
            return f"Projet '{project_name}' introuvable dans le pr_generator."
        
        vars_list = json.loads(project.variables) if project.variables else ["text"]
        result = {
            "project": project_name,
            "variables": vars_list,
            "columns_available": ["id", "content (concaténé)", "label", "created_at"] + vars_list
        }
        return json.dumps(result, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        return f"Erreur lors de la récupération du schéma : {str(e)}"
    finally:
        session.close()

@tool("query_generator_database", "Exécute une requête SQL SELECT sécurisée sur la base de données du pr_generator (tables : 'projects' avec les colonnes [id, name, variables] et 'generated_data' avec les colonnes [id, project_id, content, label, created_at, inputs]). Préfixez les tables avec le schéma 'jan' (ex: 'jan.projects', 'jan.generated_data'). Les requêtes d'écriture sont interdites.")
def query_generator_database(sql_query: str):
    """
    Permet à l'agent conversationnel d'interroger directement la base de données de pr_generator
    à l'aide d'une requête SQL SELECT (lecture seule).
    """
    query_clean = sql_query.strip()
    if not query_clean.lower().startswith("select"):
        return "Erreur : Seules les requêtes SELECT en lecture seule sont autorisées sur la base de données."

    session = factory.Session()
    try:
        # Exécution brute
        result = session.execute(text(query_clean))
        # Si la requête retourne des lignes (SELECT)
        if result.returns_rows:
            cols = result.keys()
            rows = result.fetchall()
            # Convertir les lignes en liste de dicts
            data = [dict(zip(cols, row)) for row in rows]
            # Formater joliment en JSON
            return json.dumps(data, indent=2, default=str, ensure_ascii=False)
        else:
            return "Requête exécutée avec succès (aucune ligne retournée)."
    except Exception as e:
        return f"Erreur lors de l'exécution de la requête SQL : {str(e)}"
    finally:
        session.close()

@tool("analyze_generator_data", "Récupère les données d'un projet de génération de données sous forme structurée pour analyse (limité à un nombre maximum de lignes).")
def analyze_generator_data(project_name: str, limit: int = 50):
    """
    Récupère jusqu'à 'limit' lignes de données générées pour le projet spécifié sous forme de chaîne JSON.
    Cela permet à l'agent de traiter et analyser le contenu en contexte.
    """
    session = factory.Session()
    try:
        project = session.query(factory.Project).filter_by(name=project_name).first()
        if not project:
            return f"Projet '{project_name}' introuvable."

        rows = session.query(factory.GeneratedData).filter_by(project_id=project.id).limit(limit).all()
        result = []
        for r in rows:
            inputs_dict = json.loads(r.inputs) if r.inputs else {"text": r.content}
            result.append({
                "id": r.id,
                "label": r.label,
                "inputs": inputs_dict,
                "content_preview": r.content
            })
        return json.dumps(result, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        return f"Erreur lors de la récupération des données : {str(e)}"
    finally:
        session.close()
