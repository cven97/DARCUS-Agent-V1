from jan_agent.tools import tool

@tool("saluer_utilisateur", "Salue l'utilisateur par son nom de manière chaleureuse.")
def saluer_utilisateur(nom):
    """
    Exemple d'outil personnalisé.
    Il suffit de décorer votre fonction avec @tool(nom, description)
    et de la placer dans ce dossier 'custom_tools'.
    """
    return f"Bonjour {nom} ! C'est un plaisir de t'aider aujourd'hui. Bienvenue dans vos outils personnalisés."
