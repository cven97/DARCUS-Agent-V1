import json
import sys
from pathlib import Path
from .utils import color

def load_config(path="config.json"):
    config_path = Path(path)
    if not config_path.exists():
        print(color(f"Erreur : {path} introuvable.", "red"))
        sys.exit(1)

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as exc:
        print(color(f"Erreur : config JSON invalide ({exc}).", "red"))
        sys.exit(1)

    required_sections = ["server", "model", "files", "security", "system_prompt"]
    missing = [section for section in required_sections if section not in config]
    if missing:
        print(color(f"Erreur : sections manquantes dans config.json : {', '.join(missing)}", "red"))
        sys.exit(1)
    return config
