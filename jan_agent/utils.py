import time
import re
import threading

ANSI = {
    "blue": "\033[94m",
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "purple": "\033[95m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}

LOG_LEVELS = {"non", "debug", "all"}

STOPWORDS = {
    "avec", "dans", "des", "donc", "elle", "est", "les", "mais", "par", "pas",
    "plus", "pour", "que", "qui", "sur", "une", "vous", "the", "and", "for",
    "that", "this", "not", "are", "from", "have", "your",
}


def color(text, name):
    return f"{ANSI[name]}{text}{ANSI['reset']}"


def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class AgentLogger:
    def __init__(self, level="non"):
        self.lock = threading.Lock()
        self.level = self._normalize(level)

    def _normalize(self, level):
        if isinstance(level, dict):
            level = level.get("level", "non")
        level = str(level or "non").lower()
        return level if level in LOG_LEVELS else "non"

    def set_level(self, level):
        with self.lock:
            self.level = self._normalize(level)

    def enabled(self, message_level):
        with self.lock:
            current = self.level
        if current == "non":
            return False
        if current == "debug":
            return message_level == "debug"
        return message_level in {"debug", "all"}

    def log(self, agent, message, message_level="debug"):
        if not self.enabled(message_level):
            return
        with self.lock:
            print(color(f"\n[{agent}] {message}", "purple"))


def extract_terms(text):
    words = re.findall(r"[\wÀ-ÿ]{3,}", text.lower())
    return [word for word in words if word not in STOPWORDS]


GREETING_PATTERNS = (
    "bonjour", "salut", "coucou", "bonsoir", "hello", "hi ", "hey",
    "merci", "au revoir", "bye", "à bientôt", "ça va", "ca va",
    "ok", "d'accord", "daccord", "super", "cool", "parfait",
)

CODING_KEYWORDS = (
    "```", "erreur", "exception", "stacktrace", "stack trace", "bug",
    "fonction", "méthode", "classe", "variable", "compile", "compilation",
    "code", "script", "algorithme", "api", "endpoint", "base de données",
    "requête sql", "sql", "docker", "git ", "git\n", "python", "java",
    "javascript", "typescript", "angular", "react", "spring boot",
    "fastapi", "refactor", "débogue", "debug", "optimise",
)
CODING_EXTENSIONS = (
    ".py", ".java", ".ts", ".js", ".sql", ".json", ".yml", ".yaml",
    ".xml", ".html", ".css", ".sh", ".c", ".cpp", ".go", ".rs",
)

RESEARCH_KEYWORDS = (
    "recherche sur le web", "recherche web", "cherche sur", "trouve-moi",
    "trouve moi", "dernière version", "derniere version", "actualité",
    "actualités", "météo", "meteo", "quel jour sommes", "aujourd'hui",
    "prix de", "cours de", "aujourd'hui", "2024", "2025", "2026",
)


def heuristic_classify(query):
    """
    Classification instantanée, sans appel LLM, pour les cas évidents.
    Retourne 'conversation' | 'coding' | 'research', ou None si ambigu
    (auquel cas l'appelant doit retomber sur la classification LLM).
    """
    if not query:
        return None
    text = query.strip().lower()
    if not text:
        return None

    word_count = len(text.split())

    # Signaux non ambigus en premier : un fence de code ou une extension de
    # fichier ne peut raisonnablement signifier autre chose que du code.
    if any(ext in text for ext in CODING_EXTENSIONS) or "```" in query:
        return "coding"

    # Les marqueurs explicites de recherche/actualité priment sur les simples
    # mentions de langage/framework (ex: "cherche la dernière version de
    # Angular" est une recherche, pas une demande de code).
    if any(keyword in text for keyword in RESEARCH_KEYWORDS):
        return "research"

    if any(keyword in text for keyword in CODING_KEYWORDS):
        return "coding"

    if word_count <= 6 and any(text.startswith(p) or p in text for p in GREETING_PATTERNS):
        return "conversation"

    return None


def build_local_summary(rows, max_chars=1600):
    role_names = {"user": "Utilisateur", "assistant": "Assistant", "system": "Système"}
    chunks = []
    for row in rows:
        content = " ".join(row["content"].split())
        if len(content) > 220:
            content = content[:217] + "..."
        chunks.append(f"- {role_names.get(row['role'], row['role'])}: {content}")

    summary = "Résumé automatique des anciens échanges:\n" + "\n".join(chunks)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    return summary


def call_llm(server_url, headers, model_id, messages, temperature=0.3, max_tokens=2048, timeout=60, num_ctx=8192,
             keep_alive=None, session=None):
    import requests
    http = session or requests
    try:
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "options": {"num_ctx": num_ctx}
        }
        if keep_alive is not None:
            # Extension Ollama : garde le modèle chargé en mémoire entre deux
            # appels pour éviter de payer un rechargement complet à chaque
            # message. Ignoré sans effet par les backends qui ne le supportent
            # pas (LM Studio, llama.cpp).
            payload["keep_alive"] = keep_alive
        response = http.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Erreur d'appel LLM: {e}")


def resolve_model_id(server_url, headers, config_model_name, timeout=10, session=None):
    import requests
    http = session or requests
    if not config_model_name or config_model_name == "default":
        try:
            response = http.get(f"{server_url}/v1/models", headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if data.get("data"):
                return data["data"][0]["id"]
        except Exception:
            pass
        return "default"
    return config_model_name

