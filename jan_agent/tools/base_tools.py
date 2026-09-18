import re
import requests
from pathlib import Path
from .registry import tool

@tool("read_file", "Lit le contenu d'un fichier spécifié par son chemin, avec possibilité de limiter aux lignes de start_line à end_line (1-indexées).")
def read_file(path, start_line=None, end_line=None):
    try:
        p = Path(path).expanduser()
        if not p.is_file():
            return f"Erreur : Le fichier '{path}' n'existe pas ou n'est pas un fichier."
        lines = p.read_text(encoding="utf-8").splitlines()
        
        start = 0
        if start_line is not None:
            try:
                start = max(1, int(start_line)) - 1
            except ValueError:
                pass
                
        end = len(lines)
        if end_line is not None:
            try:
                end = min(len(lines), int(end_line))
            except ValueError:
                pass
                
        return "\n".join(lines[start:end])
    except Exception as e:
        return f"Erreur de lecture du fichier : {e}"

@tool("write_file", "Crée ou écrase un fichier avec le contenu fourni.")
def write_file(path, content):
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Fichier '{path}' écrit avec succès ({len(content)} caractères)."
    except Exception as e:
        return f"Erreur d'écriture du fichier : {e}"

@tool("patch_file", "Remplace une portion de texte exacte (search) par une autre (replace) dans le fichier spécifié.")
def patch_file(path, search, replace):
    try:
        p = Path(path).expanduser()
        if not p.is_file():
            return f"Erreur : Le fichier '{path}' n'existe pas."
        content = p.read_text(encoding="utf-8")
        if search not in content:
            return f"Erreur : Le texte recherché 'search' n'a pas été trouvé dans le fichier."
            
        occurrences = content.count(search)
        if occurrences > 1:
            return f"Erreur : {occurrences} occurrences de 'search' trouvées. Le texte recherché doit être unique."
            
        new_content = content.replace(search, replace, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"Fichier '{path}' modifié avec succès."
    except Exception as e:
        return f"Erreur lors de la modification du fichier : {e}"

@tool("list_directory", "Liste les fichiers et dossiers présents dans le répertoire spécifié.")
def list_directory(path="."):
    try:
        p = Path(path).expanduser()
        if not p.is_dir():
            return f"Erreur : Le dossier '{path}' n'existe pas ou n'est pas un dossier."
        names = sorted((item.name + ("/" if item.is_dir() else "")) for item in p.iterdir())
        return "\n".join(names) if names else "(Répertoire vide)"
    except Exception as e:
        return f"Erreur de listage : {e}"

@tool("web_search", "Effectue une recherche Web pour trouver des informations récentes.")
def web_search(query):
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results_list = list(ddgs.text(query, max_results=5))

        results = []
        for i, res in enumerate(results_list):
            title = res.get("title", "Sans titre")
            url = res.get("href", res.get("url", ""))
            body = res.get("body", "")
            results.append(f"{i+1}. {title} ({url})\n   {body}")

        if not results:
            return f"Recherche Web pour '{query}' : Aucun résultat trouvé."
        return f"Résultats de recherche pour '{query}' :\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Erreur de recherche Web pour '{query}' : {e}"

@tool("run_command", "Exécute une commande système autorisée dans le terminal.")
def run_command(command):
    # L'exécution réelle et la validation utilisateur de cette commande
    # sont interceptées dans core.py pour passer par le runtime asynchrone.
    return f"run_command: {command}"
