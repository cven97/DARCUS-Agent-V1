import json
import os
import platform
import re
import shlex
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
import requests

from .utils import color, now_ts, ANSI, AgentLogger, call_llm, resolve_model_id, heuristic_classify
from .config import load_config
from .database import SQLiteStore
from .runtime import AgentRuntime
from .tools import get_tools_instruction, execute_tool, load_custom_tools
from .react_parser import (
    BlockDeltaStreamer,
    extract_fallback_response,
    extract_response,
    extract_thought,
    extract_tool_call,
    iter_stream_content,
    truncate_at_call,
)
from .agents import (
    ConversationAgent,
    CodingAgent,
    ResearchAgent,
    CriticAgent,
)


class JanAgent:
    def __init__(self, config_path="config.json", output_callback=None, input_callback=None):
        self.config_path = Path(config_path)
        self.cfg = load_config(config_path)
        self.logger = AgentLogger(self.cfg.get("log", "non"))
        self.security_lock = threading.RLock()
        
        self.output_callback = output_callback or self._default_output
        self.input_callback = input_callback or self._default_input

        # Session HTTP persistante pour réutiliser la connexion vers le serveur LLM
        # (classification, génération, critic, plan...) au lieu d'en ouvrir une par appel.
        self.session = requests.Session()

        # Charger l'URL initiale
        self.url = self.cfg["server"]["url"].rstrip("/")
        self.token = self.cfg["server"].get("token", "")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        
        # Essayer de détecter et basculer sur un serveur LLM alternatif si celui configuré est éteint
        self._detect_and_bind_server()
        
        # Appliquer le reste de la config
        self._apply_config(self.cfg)

        self.retry_count = 0
        self.store = SQLiteStore(self.database_file)
        self.store.sync_security(self.cfg["security"])
        self._load_security_from_store()
        self.store.migrate_json_history(self.history_file, self.system_prompt)
        
        # Enregistrer la config dans la DB pour les mini-agents d'arrière-plan
        self.store.put_setting("config", self.cfg)
        
        # Charger les outils personnalisés
        load_custom_tools("custom_tools")
        
        # Instancier les agents spécialisés
        self.conversation_agent = ConversationAgent()
        self.coding_agent = CodingAgent()
        self.research_agent = ResearchAgent()
        self.critic_agent = CriticAgent()

        # Enregistrer les agents spécialistes dans la base de données
        try:
            self.store.register_agent("router", "Routeur", "running")
            self.store.register_agent("planner", "Planificateur", "running")
            self.store.register_agent("conversation-agent", "Conversation & Assistance générale", "running")
            self.store.register_agent("coding-agent", "Développement logiciel & Architecture", "running")
            self.store.register_agent("research-agent", "Recherche d'informations & Web", "running")
            self.store.register_agent("critic-agent", "Audit de réponses & de code", "running")
        except Exception as e:
            self.logger.log("jan-agent", f"Erreur lors de l'enregistrement des agents : {e}", "debug")

        self.runtime = AgentRuntime(
            self.store,
            self.config_path,
            self._apply_config,
            self._execute_command,
            self.logger,
        )
        self.runtime.start()

    def _default_output(self, event_type, content):
        if event_type == "thought":
            print(f"\n{color('Pensée de l\'agent :', 'purple')} {content}")
        elif event_type == "tool_requested":
            tool_name, args = content
            print(f"\n{color('Outil demandé :', 'yellow')} {tool_name} {args}")
        elif event_type == "tool_executed":
            cmd_to_run = content
            print(color(f"[task-agent] Exécution autorisée : {cmd_to_run}", "yellow"))
        elif event_type == "observation_received":
            print(color(f"[Système] Observation enregistrée ({len(content)} caractères).", "blue"))
        elif event_type == "response":
            print(f"\n{color('Assistant :', 'green')} {content}")
        elif event_type == "security_error":
            print(color(f"[SÉCURITÉ] {content}", "red"))
        elif event_type == "info":
            print(color(f"[Système] {content}", "yellow"))

    def _default_input(self, tool_name, args):
        if tool_name == "run_command":
            cmd_to_run = args.get("command", "")
            cmd_parts = cmd_to_run.strip().split()
            cmd_base = cmd_parts[0].lower() if cmd_parts else ""
            user_choice = input(color(
                f"Autoriser l'outil 'run_command' pour '{cmd_to_run}' ?\n"
                f"  (y: oui, n: non et arrêter, a: toujours autoriser '{cmd_base}') [y] : ",
                "bold"
            )).strip().lower()
            return user_choice
        else:
            user_choice = input(color(f"Autoriser l'outil '{tool_name}' avec {args} ? (y/n) [y] : ", "bold")).strip().lower()
            return user_choice

    def _detect_and_bind_server(self):
        """
        Teste la connexion au serveur configuré et bascule vers Ollama ou LM Studio si indisponible.
        """
        try:
            response = requests.get(f"{self.url}/v1/models", headers=self.headers, timeout=2)
            if response.status_code == 200:
                return  # Serveur fonctionnel
        except Exception:
            pass

        # Fallbacks : Ollama, LM Studio/Jan, Llama.cpp
        fallbacks = [
            ("http://localhost:11434", ""),
            ("http://localhost:1234", "1234"),
            ("http://localhost:8080", ""),
        ]
        
        for fallback_url, fallback_token in fallbacks:
            if fallback_url.rstrip("/") == self.url.rstrip("/"):
                continue
            try:
                headers = {
                    "Authorization": f"Bearer {fallback_token}",
                    "Content-Type": "application/json",
                }
                response = requests.get(f"{fallback_url}/v1/models", headers=headers, timeout=2)
                if response.status_code == 200:
                    self.url = fallback_url.rstrip("/")
                    self.token = fallback_token
                    self.headers = headers
                    self.cfg["server"]["url"] = self.url
                    self.cfg["server"]["token"] = self.token
                    print(color(f"\n[Réseau] Serveur configuré inaccessible, bascule sur le fallback : {self.url}", "yellow"))
                    return
            except Exception:
                pass

    def _apply_config(self, cfg):
        self.cfg = cfg
        server_cfg = cfg["server"]
        files_cfg = cfg["files"]
        security_cfg = cfg["security"]

        self.url = server_cfg["url"].rstrip("/")
        self.token = server_cfg.get("token", "")
        self.timeout = int(server_cfg.get("timeout", 60))
        # Extension Ollama : garde le modèle chargé en mémoire entre deux appels
        # pour éviter un rechargement complet à chaque message.
        self.keep_alive = server_cfg.get("keep_alive", "30m")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        self.history_file = Path(files_cfg.get("history_file", "memory.json"))
        self.log_file = Path(files_cfg.get("log_file", "activity_log.txt"))
        self.database_file = Path(files_cfg.get("database_file", "agent_memory.sqlite3"))
        with self.security_lock:
            self.allowed = {cmd.lower().strip() for cmd in security_cfg.get("allowed_commands", []) if cmd}
            self.forbidden = [kw.lower().strip() for kw in security_cfg.get("forbidden_keywords", []) if kw]
            self.max_retry = int(security_cfg.get("max_retry_count", 1))
        self.system_prompt = cfg["system_prompt"]
        if hasattr(self, "logger"):
            self.logger.set_level(cfg.get("log", "non"))
        if hasattr(self, "store"):
            self.store.sync_security(security_cfg)
            self._load_security_from_store()
            self.store.put_setting("config", cfg)
            self.logger.log("security-agent", "règles sécurité synchronisées dans SQLite", "debug")

    def _load_security_from_store(self):
        policy = self.store.get_security_policy()
        with self.security_lock:
            self.allowed = set(policy["allowed_commands"])
            self.forbidden = list(policy["forbidden_keywords"])
            self.max_retry = policy["max_retry_count"]

    def _add_to_auto_approved(self, cmd_base):
        security_cfg = self.cfg.setdefault("security", {})
        auto_approved = security_cfg.setdefault("auto_approved_commands", [])
        if cmd_base not in auto_approved:
            auto_approved.append(cmd_base)
            try:
                self.config_path.write_text(json.dumps(self.cfg, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
                if hasattr(self, "store"):
                    self.store.put_setting("config", self.cfg)
                print(color(f"[Configuration] Commande '{cmd_base}' ajoutée de manière permanente à la whitelist d'approbation automatique.", "green"))
            except Exception as e:
                print(color(f"Erreur d'écriture dans config.json : {e}", "red"))

    def _get_security_snapshot(self):
        if hasattr(self, "store"):
            policy = self.store.get_security_policy()
            with self.security_lock:
                self.allowed = set(policy["allowed_commands"])
                self.forbidden = list(policy["forbidden_keywords"])
                self.max_retry = policy["max_retry_count"]

        with self.security_lock:
            return set(self.allowed), list(self.forbidden)

    def close(self):
        self.runtime.stop()
        self.store.close()

    def _log(self, message):
        self.store.log_event("info", "jan-agent", message)
        self.logger.log("jan-agent", message, "debug")
        try:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(f"[{now_ts()}] {message}\n")
        except OSError:
            pass

    def reset_memory(self, conversation_id=1):
        self.store.reset_conversation(conversation_id)
        print(color(f"\n[Système] Mémoire de la conversation {conversation_id} réinitialisée.", "yellow"))

    def build_messages(self, user_msg, conversation_id=1, agent_mode="loop", agent_type=None):
        memory_cfg = self.cfg.get("memory", {})
        threshold = memory_cfg.get("threshold", 28)

        # Résoudre le prompt système selon l'agent spécialiste ciblé
        system_prompt = self.system_prompt
        if agent_type == "conversation":
            system_prompt = self.conversation_agent.get_system_prompt(self.cfg)
        elif agent_type == "coding":
            system_prompt = self.coding_agent.get_system_prompt(self.cfg)
        elif agent_type == "research":
            system_prompt = self.research_agent.get_system_prompt(self.cfg)

        if agent_mode == "generative":
            system_content = system_prompt
            summary = self.store.get_latest_summary(conversation_id=conversation_id)
            if summary:
                system_content += f"\n\nCONTEXTE HISTORIQUE CONSOLIDÉ :\n{summary}"
            retrieved = self.store.retrieve_context(user_msg, conversation_id=conversation_id, limit=3)
            if retrieved:
                context = "\n".join(f"- {item['role']}: {item['content']}" for item in retrieved)
                system_content += f"\n\nSOUVENIRS LIÉS À LA REQUÊTE :\n{context}"
            messages = [{"role": "system", "content": system_content}]
            raw_history = self.store.get_recent_messages(limit=threshold, conversation_id=conversation_id)
            for msg in raw_history:
                role = msg["role"]
                if role == "system":
                    role = "user"
                messages.append({"role": role, "content": msg["content"]})
            return messages

        tools_instruction = get_tools_instruction()
        
        # Injection du plan si existant
        plan_injection = ""
        current_step = None
        active_plan = self.store.get_active_plan(conversation_id)
        if active_plan:
            plan_injection = f"PLAN D'ACTION ACTUEL (Objectif : {active_plan['query']})\n"
            current_step = None
            for step in active_plan["steps"]:
                status_icon = "✅" if step["status"] == "done" else "🔄" if step["status"] == "in_progress" else "⏳"
                result_text = f" (Résultat: {step['result_summary']})" if step['result_summary'] else ""
                plan_injection += f"- Étape {step['step_number']} {status_icon} : {step['description']}{result_text}\n"
                if step["status"] == "pending" and not current_step:
                    current_step = step
            
            if current_step:
                self.store.update_plan_step(current_step["id"], "in_progress")
                plan_injection += f"\nATTENTION : Ton UNIQUE but pour le moment est de résoudre l'Étape {current_step['step_number']} ({current_step['description']}). "
                plan_injection += "Dès que tu as fini cette étape, utilise l'outil `complete_current_step` avec un résumé de ce que tu as trouvé. Ne fais rien d'autre.\n\n"
            else:
                self.store.mark_plan_completed(active_plan["id"])
                plan_injection += "\nToutes les étapes du plan sont terminées ! Tu peux maintenant synthétiser et formuler ta <response> finale à l'utilisateur.\n\n"

        if current_step:
            coordination_prompt = (
                f"{system_prompt}\n\n"
                f"{plan_injection}"
                "RÈGLES DE FORMATAGE XML DE LA BOUCLE DE RAISONNEMENT :\n"
                "Tu dois impérativement formater tes réponses avec la structure suivante :\n\n"
                "<thought>\n"
                "Raisonnement...\n"
                "</thought>\n"
                "<call name=\"nom_de_l_outil\">\n"
                "{\"nom_argument\": \"valeur_argument\"}\n"
                "</call>\n\n"
                "ATTENTION TRÈS IMPORTANTE : Tu n'as PAS le droit de formuler de réponse finale à l'utilisateur pour le moment. "
                "Tu dois UNIQUEMENT accomplir l'étape en cours en utilisant les outils nécessaires, puis appeler l'outil `complete_current_step` lorsque tu as terminé l'étape.\n\n"
                f"{tools_instruction}\n"
                "OUTILS DE PLANIFICATION DISPONIBLES :\n"
                "- complete_current_step: {\"summary\": \"Résumé des trouvailles pour cette étape\"}\n"
            )
        else:
            coordination_prompt = (
                f"{system_prompt}\n\n"
                f"{plan_injection}"
                "RÈGLES DE FORMATAGE XML DE LA BOUCLE DE RAISONNEMENT (ReAct) :\n"
                "Tu dois impérativement formater tes réponses selon l'une des deux structures ci-dessous, sans aucun texte libre en dehors de ces balises.\n\n"
                "Structure 1 : Réfléchir et appeler un outil\n"
                "<thought>\n"
                "Raisonnement expliquant pourquoi tu as besoin de cet outil pour avancer dans la tâche.\n"
                "</thought>\n"
                "<call name=\"nom_de_l_outil\">\n"
                "{\"nom_argument\": \"valeur_argument\"}\n"
                "</call>\n\n"
                "Structure 2 : Donner la réponse finale\n"
                "<thought>\n"
                "Explication montrant que tu as toutes les informations requises pour clore la tâche.\n"
                "</thought>\n"
                "<response>\n"
                "Réponse finale rédigée de manière claire et concise à destination de l'utilisateur.\n"
                "</response>\n\n"
                "ATTENTION TRÈS IMPORTANTE : Tes connaissances internes peuvent être obsolètes. Dès que tu appelles un outil (comme une recherche web), tu DOIS formuler ta <response> finale en te basant en priorité sur les informations retournées dans l'Observation Système. Si l'outil ne retourne aucun résultat, tu es autorisé à utiliser tes connaissances internes pour répondre.\n"
                "RÈGLE ANTI-BOUCLE : Si une observation te donne l'information suffisante, donne IMMÉDIATEMENT ta réponse dans <response>...</response>. Ne cherche pas à être exhaustif. Ne répète JAMAIS une recherche ou un outil similaire.\n"
                "RÈGLE MESSAGES SIMPLES : Pour les salutations (salut, bonjour, merci, etc.) et les questions générales de conversation qui ne nécessitent PAS d'outil, réponds DIRECTEMENT avec <response>...</response> SANS appeler d'outil. N'invente pas d'outils qui n'existent pas (comme 'wait_for_user_input').\n\n"
                f"{tools_instruction}\n"
            )
            
            # Ajouter l'outil create_plan si le mode est tool et aucun plan actif
            if self.cfg.get("model", {}).get("planning_mode") == "tool" and not active_plan:
                coordination_prompt += "- create_plan: {\"query\": \"Le but global pour lequel générer un plan\"}\n"
        
        system_content = coordination_prompt
        
        summary = self.store.get_latest_summary(conversation_id=conversation_id)
        if summary:
            system_content += f"\n\nCONTEXTE HISTORIQUE CONSOLIDÉ :\n{summary}"
            self.logger.log("memory-agent", "résumé SQLite injecté dans le contexte", "all")

        retrieved = self.store.retrieve_context(user_msg, conversation_id=conversation_id, limit=3)
        if retrieved:
            context = "\n".join(f"- {item['role']}: {item['content']}" for item in retrieved)
            system_content += f"\n\nSOUVENIRS LIÉS À LA REQUÊTE :\n{context}"
            self.logger.log("retrieval-agent", f"{len(retrieved)} souvenir(s) injecté(s)", "debug")

        messages = [{"role": "system", "content": system_content}]
        raw_history = self.store.get_recent_messages(limit=threshold, conversation_id=conversation_id)
        for msg in raw_history:
            role = msg["role"]
            if role == "system":
                role = "user"
            messages.append({"role": role, "content": msg["content"]})
        return messages

    def _split_command(self, cmd):
        try:
            return shlex.split(cmd, posix=os.name != "nt"), None
        except ValueError as exc:
            return None, f"Commande invalide : {exc}"

    def _normalize_command(self, args):
        if not args:
            return args

        base = args[0].lower()
        is_windows = platform.system().lower() == "windows"
        translations = {"ls": "dir", "cat": "type"} if is_windows else {"dir": "ls", "type": "cat"}
        args[0] = translations.get(base, base)
        return args

    def _is_safe(self, raw_cmd, args):
        low = raw_cmd.lower().strip()
        if not low:
            return False, "Commande vide."

        allowed, forbidden = self._get_security_snapshot()

        for keyword in forbidden:
            if keyword and keyword in low:
                return False, f"Mot-clé interdit : {keyword}"

        if any(token in low for token in [";", "&&", "||", "|", "`", "$(", "<"]):
            return False, "Opérateur shell interdit."

        raw_args, parse_error = self._split_command(raw_cmd)
        if parse_error:
            return False, parse_error

        raw_base = raw_args[0].lower() if raw_args else ""
        normalized_base = args[0].lower() if args else ""
        if raw_base not in allowed and normalized_base not in allowed:
            visible_base = raw_base or normalized_base
            return False, f"Commande non autorisée : {visible_base}"
        return True, ""

    def _write_echo_redirection(self, args):
        if ">" not in args:
            return None

        if args[0].lower() != "echo":
            return 1, "", "Seule la redirection 'echo ... > fichier' est prise en charge."

        redirect_index = args.index(">")
        content = " ".join(args[1:redirect_index])
        targets = args[redirect_index + 1:]
        if len(targets) != 1:
            return 1, "", "La redirection doit cibler un seul fichier."

        target = Path(targets[0]).expanduser()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content + "\n", encoding="utf-8")
        except OSError as exc:
            return 1, "", f"Erreur écriture : {exc}"
        return 0, f"Fichier écrit : {target}", ""

    def _run_builtin(self, args):
        base = args[0].lower()
        try:
            if base == "cd":
                if len(args) != 2:
                    return 1, "", "Usage : cd <dossier>"
                os.chdir(Path(args[1]).expanduser())
                return 0, f"Nouveau répertoire : {os.getcwd()}", ""

            if base == "pwd":
                return 0, os.getcwd(), ""

            if base in {"ls", "dir"}:
                target = Path(args[1]).expanduser() if len(args) > 1 else Path.cwd()
                names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
                return 0, "\n".join(names), ""

            if base in {"cat", "type"}:
                if len(args) != 2:
                    return 1, "", f"Usage : {base} <fichier>"
                return 0, Path(args[1]).expanduser().read_text(encoding="utf-8"), ""

            if base == "mkdir":
                if len(args) < 2:
                    return 1, "", "Usage : mkdir <dossier>"
                for path in args[1:]:
                    Path(path).expanduser().mkdir(parents=True, exist_ok=True)
                return 0, "Dossier(s) créé(s).", ""

            if base == "echo":
                return 0, " ".join(args[1:]), ""
        except OSError as exc:
            return 1, "", str(exc)
        return None

    def _execute_command(self, raw_cmd):
        import subprocess # Local import to run command

        args, error = self._split_command(raw_cmd)
        if error:
            return 1, "", error

        args = self._normalize_command(args)
        safe, reason = self._is_safe(raw_cmd, args)
        if not safe:
            return 126, "", reason

        redirected = self._write_echo_redirection(args)
        if redirected is not None:
            return redirected

        builtin_result = self._run_builtin(args)
        if builtin_result is not None:
            return builtin_result

        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError:
            return 127, "", f"Programme introuvable : {args[0]}"
        except subprocess.TimeoutExpired:
            return 124, "", f"Commande interrompue après {self.timeout}s."

        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()

    def _animate(self, stop_ev):
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while not stop_ev.is_set():
            sys.stdout.write(f"\r{color(frames[i % len(frames)] + ' Jan réfléchit...', 'blue')}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()

    def list_models(self):
        try:
            response = requests.get(f"{self.url}/v1/models", headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            print(color(f"Erreur réseau Jan : {exc}", "red"))
            return None
        except ValueError:
            print(color("Erreur : réponse Jan non JSON.", "red"))
            return None

        if not data.get("data"):
            print(color("Erreur : aucun modèle disponible.", "red"))
            return None
        return data

    def _classify_query(self, query, model_id, timeout=300, out=None):
        """
        Appelle le LLM pour classifier la requête de l'utilisateur.
        Retourne l'un des types suivants: 'coding', 'research', 'conversation'.
        """
        out = out or self.output_callback

        # Court-circuit rapide sans appel LLM pour les cas évidents (salutations,
        # demandes de code explicites, recherches web explicites) : évite un
        # aller-retour LLM complet avant même de démarrer la vraie réponse.
        heuristic_type = heuristic_classify(query)
        if heuristic_type:
            out("info", f"Requête classifiée comme '{heuristic_type}' (heuristique, sans appel LLM)")
            return heuristic_type

        system_prompt = (
            "Tu es le routeur d'un système multi-agents.\n"
            "Analyse la requête de l'utilisateur et détermine quel agent est le plus qualifié pour y répondre.\n"
            "RÈGLES :\n"
            "1. Réponds UNIQUEMENT avec un objet JSON au format exact suivant :\n"
            "   {\"agent\": \"coding\" | \"research\" | \"conversation\", \"confidence\": float}\n"
            "2. Ne mets aucun texte avant ou après le JSON.\n"
            "3. Guide de classification :\n"
            "   - 'coding' : Requêtes concernant le développement, l'écriture de code, la correction de bugs, l'architecture logicielle, les bases de données, Docker, Git.\n"
            "   - 'research' : Requêtes nécessitant des informations récentes du web, des recherches de documentation ou de projets, ou l'utilisation d'outils de recherche.\n"
            "   - 'conversation' : Requêtes simples, salutations, remerciements, discussions générales, rédactions de textes simples ou de rapports sans code."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Requête utilisateur : \"{query}\""}
        ]
        
        try:
            out("info", "Classification de la requête en cours...")
            res = call_llm(
                server_url=self.url,
                headers=self.headers,
                model_id=model_id,
                messages=messages,
                temperature=0.0,
                max_tokens=60,
                timeout=timeout,
                num_ctx=self.cfg["model"].get("num_ctx", 8192),
                keep_alive=self.keep_alive,
                session=self.session,
            ).strip()
            
            match = re.search(r"\{.*?\}", res, re.DOTALL)
            if match:
                res = match.group(0)
            
            data = json.loads(res)
            agent_type = data.get("agent", "conversation")
            confidence = data.get("confidence", 1.0)
            out("info", f"Requête classifiée comme '{agent_type}' (confiance : {confidence:.2f})")
            return agent_type
        except Exception as e:
            self.logger.log("router", f"Échec de classification : {e}. Utilisation du fallback par défaut.", "debug")
            out("info", "Échec de classification. Redirection vers conversation par défaut.")
            return "conversation"

    @contextmanager
    def _thinking_loader(self, out):
        """Affiche l'animation de la CLI pendant un appel LLM, seulement si `out` est la sortie console par défaut."""
        stop_ev = threading.Event()
        loader = None
        if out == self._default_output:
            loader = threading.Thread(target=self._animate, args=(stop_ev,), daemon=True)
            loader.start()
        try:
            yield
        finally:
            stop_ev.set()
            if loader:
                loader.join(timeout=1)

    def _request_completion_stream(self, model_id, messages):
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": self.cfg["model"].get("temperature", 0.3),
            "max_tokens": self.cfg["model"].get("max_tokens", 2048),
            "options": {"num_ctx": self.cfg["model"].get("num_ctx", 8192)},
            "stream": True,
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        response = self.session.post(
            f"{self.url}/v1/chat/completions",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()
        return response

    def generate_completion(self, model_id, user_msg, max_steps=10, conversation_id=1, agent_mode="loop",
                             output_callback=None, input_callback=None):
        """
        Callbacks passés explicitement (plutôt que lus depuis self.output_callback/self.input_callback)
        pour que plusieurs sessions (API SSE, WebSocket, CLI) puissent utiliser la même instance
        JanAgent sans se marcher dessus. À défaut, retombe sur les callbacks par défaut de l'instance.
        """
        out = output_callback or self.output_callback
        ask = input_callback or self.input_callback

        # Classer la requête utilisateur pour choisir l'agent spécialiste adéquat
        agent_type = self._classify_query(user_msg, model_id, out=out)

        # Enregistrer le message de l'utilisateur dans l'historique SQLite
        self.store.add_message("user", user_msg, conversation_id=conversation_id)

        if agent_mode == "generative":
            self._run_generative_completion(model_id, user_msg, conversation_id, agent_type, out)
            return

        self._run_loop_completion(model_id, user_msg, max_steps, conversation_id, agent_type, out, ask)

    def _run_generative_completion(self, model_id, user_msg, conversation_id, agent_type, out):
        messages = self.build_messages(user_msg, conversation_id=conversation_id, agent_mode="generative", agent_type=agent_type)

        try:
            with self._thinking_loader(out):
                start = time.perf_counter()
                response = self._request_completion_stream(model_id, messages)

                answer = ""
                for chunk_text in iter_stream_content(response):
                    answer += chunk_text
                    out("response_delta", chunk_text)

                    had_explicit_close = "</call>" in answer
                    answer, closed = truncate_at_call(answer)
                    if closed:
                        if not had_explicit_close:
                            out("response", answer)
                        break

                self.store.put_metric("last_completion_ms", (time.perf_counter() - start) * 1000)
        except Exception as exc:
            out("security_error", f"Erreur réseau/API : {exc}")
            return

        # Enregistrer la réponse brute de l'assistant dans la DB
        self.store.add_message("assistant", answer, conversation_id=conversation_id)
        out("response", answer.strip())

    def _run_loop_completion(self, model_id, user_msg, max_steps, conversation_id, agent_type, out, ask):
        planning_mode = self.cfg.get("model", {}).get("planning_mode", "auto")

        # Auto-planning mode
        if planning_mode == "auto" and not self.store.get_active_plan(conversation_id):
            from .planner import generate_plan_steps
            out("info", "Génération automatique d'un plan de travail...")
            steps = generate_plan_steps(
                user_msg,
                self.url,
                self.headers,
                model_id,
                timeout=min(self.timeout, 120),
                planner_prompt=self.cfg.get("planner_prompt"),
                num_ctx=self.cfg["model"].get("num_ctx", 8192),
                keep_alive=self.keep_alive,
                session=self.session,
            )
            if steps:
                self.store.create_plan(conversation_id, user_msg, steps)
                out("info", f"Plan généré avec {len(steps)} étapes.")

        tool_call_history = []
        unknown_tool_errors = 0
        step = 0
        while step < max_steps:
            # Récupérer l'historique complet (y compris les pensées/observations précédentes de cette boucle)
            messages = self.build_messages(user_msg, conversation_id=conversation_id, agent_type=agent_type)

            try:
                with self._thinking_loader(out):
                    start = time.perf_counter()
                    response = self._request_completion_stream(model_id, messages)

                    answer = ""
                    deltas = BlockDeltaStreamer(out)
                    for chunk_text in iter_stream_content(response):
                        answer += chunk_text
                        answer, closed = truncate_at_call(answer)
                        if closed:
                            break
                        deltas.feed(answer)

                    self.store.put_metric("last_completion_ms", (time.perf_counter() - start) * 1000)
            except Exception as exc:
                out("security_error", f"Erreur réseau/API : Le serveur Jan local ne répond pas ou a expiré ({exc}). Veuillez vérifier qu'il tourne sur {self.url}.")
                return

            # Enregistrer la réponse brute de l'assistant dans la DB
            self.store.add_message("assistant", answer, conversation_id=conversation_id)

            # Afficher la pensée si elle est présente
            thought = extract_thought(answer)
            if thought:
                out("thought", thought)

            from jan_agent.tools.registry import get_registered_tools
            registered = get_registered_tools()
            tool_name, args = extract_tool_call(answer, registered)
            response_text = extract_response(answer)

            if tool_name:
                # Détection de boucle / appels d'outils répétés
                args_serialized = json.dumps(args, sort_keys=True)
                consecutive_count = 0
                for prev_name, prev_args_str in reversed(tool_call_history):
                    if prev_name == tool_name and prev_args_str == args_serialized:
                        consecutive_count += 1
                    else:
                        break

                tool_call_history.append((tool_name, args_serialized))

                if consecutive_count >= 1:
                    # Répéteur détecté !
                    if consecutive_count >= self.max_retry:
                        # Répété max_retry fois (ou plus), arrêt de sécurité
                        out("security_error", f"Détection de boucle infinie pour l'outil '{tool_name}' avec les mêmes arguments. Arrêt de sécurité.")
                        self.store.add_message("system", f"Erreur : L'agent a été arrêté car il bouclait sur l'appel d'outil '{tool_name}' avec les mêmes arguments.", conversation_id=conversation_id)
                        break

                    # Première répétition : injecter un avertissement comme observation
                    warning_msg = (
                        f"Erreur : Tu as déjà appelé l'outil '{tool_name}' avec ces paramètres exacts à l'étape précédente.\n"
                        f"L'observation système n'a pas changé. Si le résultat précédent n'était pas suffisant, ne répète pas le même appel.\n"
                        f"Tu DOIS essayer une approche différente (comme changer d'arguments) ou formuler ta <response> finale si tu as assez d'informations."
                    )
                    self.store.add_message("system", f"<observation>\n{warning_msg}\n</observation>", conversation_id=conversation_id)
                    out("observation_received", warning_msg)
                    step += 1
                    continue

                out("tool_requested", (tool_name, args))

                # Validation de sécurité interactive (demandée par l'utilisateur)
                confirm = True
                if tool_name == "run_command":
                    cmd_to_run = args.get("command", "")
                    cmd_parts = cmd_to_run.strip().split()
                    cmd_base = cmd_parts[0].lower() if cmd_parts else ""

                    # Récupérer la liste d'auto-approbation
                    auto_approved = self.cfg.get("security", {}).get("auto_approved_commands", [])

                    if cmd_base in auto_approved:
                        out("info", f"[Auto-Approbation] Exécution automatique de '{cmd_to_run}' (commande '{cmd_base}' auto-approuvée).")
                    else:
                        user_choice = ask(tool_name, args)
                        if user_choice == "a":
                            self._add_to_auto_approved(cmd_base)
                        elif user_choice in ["", "y", "yes", "oui"]:
                            pass
                        else:
                            confirm = False
                            out("security_error", "Action annulée. Arrêt immédiat de l'agent.")
                            self.store.add_message("system", "Erreur : Exécution refusée par l'utilisateur. La tâche a été abandonnée.", conversation_id=conversation_id)

                elif tool_name in ["write_file", "patch_file"]:
                    user_choice = ask(tool_name, args)
                    if user_choice not in ["", "y", "yes", "oui"]:
                        confirm = False
                        out("security_error", "Action annulée. Arrêt immédiat de l'agent.")
                        self.store.add_message("system", "Erreur : Exécution refusée par l'utilisateur. La tâche a été abandonnée.", conversation_id=conversation_id)

                if not confirm:
                    break

                if tool_name == "complete_current_step":
                    summary = args.get("summary", "")
                    active_plan = self.store.get_active_plan(conversation_id)
                    if active_plan:
                        for pstep in active_plan["steps"]:
                            if pstep["status"] == "in_progress":
                                self.store.update_plan_step(pstep["id"], "done", summary)
                                observation = f"Étape {pstep['step_number']} complétée avec succès. Résultat sauvegardé : {summary}"
                                break
                        else:
                            observation = "Aucune étape en cours trouvée."
                    else:
                        observation = "Aucun plan actif."
                elif tool_name == "create_plan":
                    query = args.get("query", user_msg)
                    from .planner import generate_plan_steps
                    out("info", "Création d'un plan de travail à la demande de l'agent...")
                    steps = generate_plan_steps(
                        query, self.url, self.headers, model_id,
                        timeout=min(self.timeout, 120),
                        num_ctx=self.cfg["model"].get("num_ctx", 8192),
                        keep_alive=self.keep_alive,
                        session=self.session,
                    )
                    if steps:
                        self.store.create_plan(conversation_id, query, steps)
                        observation = f"Plan créé avec {len(steps)} étapes. Le plan sera injecté dans ton prompt à la prochaine étape."
                    else:
                        observation = "Erreur lors de la génération du plan."
                elif tool_name == "run_command":
                    cmd_to_run = args.get("command", "")
                    out("tool_executed", cmd_to_run)
                    task_result = self.runtime.submit_command(cmd_to_run, timeout=self.timeout + 5)
                    if task_result["status"] == "done":
                        payload = json.loads(task_result["result"])
                        code = payload["code"]
                        observation = payload["stdout"] if code == 0 else payload["stderr"]
                    else:
                        observation = task_result.get("error") or "Erreur inconnue."
                else:
                    observation = execute_tool(tool_name, args)
                    if not isinstance(observation, str):
                        try:
                            observation = json.dumps(observation, ensure_ascii=False, indent=2)
                        except Exception:
                            observation = str(observation)
                    # Compter les erreurs d'outils inconnus
                    if observation.startswith("Erreur : Outil") and "inconnu" in observation:
                        unknown_tool_errors += 1
                        if unknown_tool_errors >= 2:
                            # Trop d'erreurs d'outils inconnus → forcer une réponse
                            force_msg = (
                                "ARRÊT OBLIGATOIRE : Tu as appelé plusieurs outils inexistants. "
                                "Tu DOIS formuler ta réponse finale MAINTENANT dans <response>...</response>. "
                                "N'appelle plus aucun outil."
                            )
                            self.store.add_message("system", f"<observation>\n{force_msg}\n</observation>", conversation_id=conversation_id)
                            out("observation_received", force_msg)
                            step += 1
                            continue
                    else:
                        unknown_tool_errors = 0  # Réinitialiser si outil valide

                # Ajouter l'observation en tant que message système
                self.store.add_message("system", f"<observation>\n{observation}\n</observation>", conversation_id=conversation_id)
                out("observation_received", observation)

            elif response_text is not None:
                final_resp = response_text

                # Validation critique pour le code
                if agent_type == "coding":
                    out("info", "[critic-agent] Analyse critique du code généré...")
                    critic_system = self.critic_agent.get_system_prompt(self.cfg)
                    critic_messages = [
                        {"role": "system", "content": critic_system},
                        {"role": "user", "content": f"Requête de l'utilisateur : {user_msg}\n\nCode proposé :\n```\n{final_resp}\n```\n\nFais une analyse critique rapide. Si tout est correct et sécurisé, réponds par 'CORRECT'. Sinon, liste précisément les problèmes ou failles de sécurité."}
                    ]
                    try:
                        critic_feedback = call_llm(
                            server_url=self.url,
                            headers=self.headers,
                            model_id=model_id,
                            messages=critic_messages,
                            temperature=0.2,
                            max_tokens=512,
                            timeout=120,
                            num_ctx=self.cfg["model"].get("num_ctx", 8192),
                            keep_alive=self.keep_alive,
                            session=self.session,
                        ).strip()

                        if "CORRECT" not in critic_feedback.upper():
                            out("thought", f"[critic-agent] Feedback critique reçu :\n{critic_feedback}")
                            warning_msg = (
                                f"[Audit Sécurité / Qualité] Le critique a détecté les problèmes suivants :\n{critic_feedback}\n"
                                f"Tu DOIS corriger ces points avant de formuler ta réponse finale."
                            )
                            self.store.add_message("system", f"<observation>\n{warning_msg}\n</observation>", conversation_id=conversation_id)
                            out("observation_received", warning_msg)
                            step += 1
                            continue
                    except Exception as critic_err:
                        self.logger.log("critic", f"Échec de l'audit critique : {critic_err}", "debug")

                out("response", final_resp)
                break
            else:
                # Notre nouveau fallback robuste en cas d'absence de balise
                final_resp = extract_fallback_response(answer)
                out("response", final_resp)
                break

            step += 1
            if step >= max_steps:
                out("info", "Nombre maximum d'étapes de raisonnement atteint.")


def main():
    agent = JanAgent()
    try:
        # Résolution du modèle par défaut
        default_cfg_model = agent.cfg.get("model", {}).get("default", "default")
        model_id = resolve_model_id(agent.url, agent.headers, default_cfg_model, timeout=5)
        
        print(color(f"--- Agent avancé prêt (Modèle: {model_id}) ---", "green"))
        print(color(f"--- Serveur actif sur : {agent.url} ---", "green"))
        print(color("--- Boucle ReAct autonome & outils actifs ---", "blue"))

        while True:
            try:
                user_input = input(f"{ANSI['bold']}Vous :{ANSI['reset']} ").strip()
            except KeyboardInterrupt:
                print()
                return 0

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "quitter"}:
                return 0
            if user_input.lower() == "reset":
                agent.reset_memory()
                continue

            agent.generate_completion(model_id, user_input)
    finally:
        agent.close()
