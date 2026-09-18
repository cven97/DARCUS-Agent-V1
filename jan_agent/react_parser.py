"""
Parsing de la boucle ReAct : détection des balises <thought>/<call>/<response>
dans les réponses streamées du LLM, et extraction des appels d'outils.
"""

import json
import re

CALL_OPEN_RE = re.compile(r"<call\s+name=['\"]?([a-zA-Z0-9_\-]+)['\"]?\s*>")
CALL_FULL_RE = re.compile(r"<call\s+name=['\"]?([a-zA-Z0-9_\-]+)['\"]?\s*>(.*?)</call>", re.DOTALL)
CALL_OPEN_TAIL_RE = re.compile(r"<call\s+name=['\"]?([a-zA-Z0-9_\-]+)['\"]?\s*>(.*)$", re.DOTALL)
CALL_INLINE_RE = re.compile(r"<call\s+name=['\"]?([a-zA-Z0-9_\-]+)['\"]?\s+([^>]+)(?:>|$)", re.DOTALL)
CMD_PATTERN = re.compile(r"\[CMD:\s*(.*?)\]", re.DOTALL)

# Commandes système que le modèle appelle parfois directement comme outils :
# elles sont réécrites en appels à run_command.
SYSTEM_COMMAND_NAMES = ("curl", "git", "python", "python3", "pip", "ls", "cat", "find", "mkdir", "pwd", "ping")

# Clé d'argument implicite quand le modèle fournit une valeur brute au lieu d'un JSON.
SINGLE_ARG_KEYS = {
    "run_command": "command",
    "list_directory": "path",
    "read_file": "path",
    "web_search": "query",
}


def iter_stream_content(response):
    """Itère sur les fragments de texte d'une réponse SSE /chat/completions."""
    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8").strip()
        if not line_str.startswith("data: "):
            continue
        data_content = line_str[6:]
        if data_content == "[DONE]":
            break
        try:
            chunk_data = json.loads(data_content)
            delta = chunk_data["choices"][0].get("delta", {})
        except Exception:
            continue
        content = delta.get("content")
        if content:
            yield content


def truncate_at_call(answer):
    """
    Coupe la réponse dès qu'un appel d'outil est complet, pour arrêter le stream
    côté client sans attendre la fin de la génération.
    Retourne (answer, True) si un appel complet a été détecté, sinon (answer, False).
    """
    if "</call>" in answer:
        idx = answer.find("</call>")
        return answer[:idx + 7], True

    # Détection proactive dès que le JSON d'arguments est refermé
    match = CALL_OPEN_RE.search(answer)
    if match:
        tag_end = match.end()
        args_str = answer[tag_end:]
        brace_count = 0
        has_json_started = False
        for i, char in enumerate(args_str):
            if char == "{":
                brace_count += 1
                has_json_started = True
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and has_json_started:
                    return answer[:tag_end + i + 1] + "</call>", True
    return answer, False


def extract_thought(answer):
    """Extrait le contenu du bloc <thought>, ou None."""
    match = re.search(r"<thought>(.*?)</thought>", answer, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_response(answer):
    """Extrait la réponse finale <response>, avec repli si la balise fermante manque."""
    match = re.search(r"<response>(.*?)</response>", answer, re.DOTALL)
    if not match:
        match = re.search(r"<response>(.*)$", answer, re.DOTALL)
    if not match:
        return None
    text = match.group(1).strip()
    if text.endswith("</response>"):
        text = text[:-len("</response>")].strip()
    return text


def extract_fallback_response(answer):
    """Dernier recours quand aucune balise attendue n'est présente : renvoie le texte utile."""
    text = re.sub(r"<thought>.*?</thought>", "", answer, flags=re.DOTALL).strip()
    text = re.sub(r"</?thought>", "", text).strip()
    if text:
        return text
    thought = extract_thought(answer)
    if thought:
        return thought
    return re.sub(r"<[^>]*>", "", answer).strip()


def extract_tool_call(answer, registered_tools):
    """
    Extrait un appel d'outil de la réponse.
    Retourne (tool_name, args) ou (None, None). Gère les formats dégradés :
    balise fermante absente, JSON dans la balise ouvrante, commande système
    appelée comme outil, arguments non-JSON, et l'ancien format [CMD: ...].
    """
    call_match = CALL_FULL_RE.search(answer)
    if not call_match:
        call_match = CALL_OPEN_TAIL_RE.search(answer)

    inline_call_match = None
    if not call_match or not call_match.group(1):
        inline_call_match = CALL_INLINE_RE.search(answer)

    if call_match or inline_call_match:
        if inline_call_match:
            tool_name = inline_call_match.group(1).strip()
            raw_args = inline_call_match.group(2).strip()
        else:
            tool_name = call_match.group(1).strip()
            raw_args = call_match.group(2).strip()
            if raw_args.endswith("</call>"):
                raw_args = raw_args[:-len("</call>")].strip()

        if tool_name not in registered_tools and tool_name.lower() in SYSTEM_COMMAND_NAMES:
            cleaned_args = raw_args.strip("\"' \t\n\r")
            raw_args = f"{tool_name} {cleaned_args}".strip()
            tool_name = "run_command"

        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            cleaned_raw = raw_args.strip("\"' \t\n\r")
            key = SINGLE_ARG_KEYS.get(tool_name)
            args = {key: cleaned_raw} if key else {}
        return tool_name, args

    cmd_match = CMD_PATTERN.search(answer)
    if cmd_match:
        return "run_command", {"command": cmd_match.group(1).strip()}
    return None, None


class BlockDeltaStreamer:
    """
    Suit le bloc actif (<thought> ou <response>) pendant le stream et émet
    uniquement les nouveaux caractères, pour éviter doublons et oscillations
    côté frontend.
    """

    def __init__(self, emit):
        self.emit = emit
        self._thought_len = 0
        self._response_len = 0

    def feed(self, answer):
        is_in_thought = "<thought>" in answer and "</thought>" not in answer
        is_in_response = "<response>" in answer and "</response>" not in answer

        if is_in_thought:
            match = re.search(r"<thought>(.*)$", answer, re.DOTALL)
            if match:
                current = match.group(1)
                new_chars = current[self._thought_len:]
                if new_chars:
                    self.emit("thought_delta", new_chars)
                    self._thought_len = len(current)
        elif is_in_response:
            match = re.search(r"<response>(.*)$", answer, re.DOTALL)
            if match:
                current = match.group(1)
                new_chars = current[self._response_len:]
                if new_chars:
                    self.emit("response_delta", new_chars)
                    self._response_len = len(current)
