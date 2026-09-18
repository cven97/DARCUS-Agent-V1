from .base import BackgroundAgent
from ..utils import build_local_summary, extract_terms, call_llm, resolve_model_id

class MemoryAgent(BackgroundAgent):
    def __init__(self, store, stop_event, logger, keep_last=18, threshold=28):
        super().__init__("memory-agent", store, stop_event, logger, interval=5)
        self.keep_last = keep_last
        self.threshold = threshold

    def tick(self):
        # Résolution de la configuration et du modèle
        cfg = self.store.get_setting("config", {})
        memory_cfg = cfg.get("memory", {})
        keep_last = memory_cfg.get("keep_last", self.keep_last)
        threshold = memory_cfg.get("threshold", self.threshold)

        conv_ids = self.store.get_conversations_needing_summary(threshold)
        if not conv_ids:
            return
            
        server_cfg = cfg.get("server", {})
        url = server_cfg.get("url", "http://localhost:1234").rstrip("/")
        token = server_cfg.get("token", "")
        timeout = int(server_cfg.get("timeout", 300))
        keep_alive = server_cfg.get("keep_alive", "30m")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        
        model_cfg = cfg.get("model", {})
        agent_model_name = model_cfg.get("memory_agent") or model_cfg.get("default") or "default"
        
        try:
            model_id = resolve_model_id(url, headers, agent_model_name, timeout=5)
        except Exception as exc:
            self.logger.log(self.name, f"Erreur de résolution du modèle : {exc}", "debug")
            return

        for conv_id in conv_ids:
            active_messages = self.store.count_active_messages(conversation_id=conv_id)
            self.logger.log(self.name, f"Conv {conv_id} : mémoire active : {active_messages} messages", "all")
            
            rows = self.store.get_messages_to_summarize(keep_last, conversation_id=conv_id)
            if not rows:
                continue
                
            try:
                # Préparation des messages pour le LLM de résumé
                role_names = {"user": "Utilisateur", "assistant": "Assistant", "system": "Système"}
                history_text = "\n".join(f"- {role_names.get(row['role'], row['role'])}: {row['content']}" for row in rows)
                
                prompt = cfg.get(
                    "memory_prompt",
                    "Fais un résumé logique condensé des échanges suivants. Conserve uniquement les décisions prises, les chemins de fichiers mentionnés et les résultats des tâches d'exécution. Sois concis."
                )
                
                # Récupère l'ancien résumé s'il existe pour faire un résumé incrémental
                old_summary = self.store.get_latest_summary(conversation_id=conv_id)
                if old_summary:
                    prompt += f"\n\nVoici le résumé précédent à intégrer :\n{old_summary}"
                
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": history_text}
                ]
                
                summary = call_llm(
                    server_url=url,
                    headers=headers,
                    model_id=model_id,
                    messages=messages,
                    temperature=model_cfg.get("temperature", 0.3),
                    max_tokens=256,
                    timeout=timeout,
                    num_ctx=model_cfg.get("num_ctx", 8192),
                    keep_alive=keep_alive,
                ).strip()
                
                if not summary:
                    summary = build_local_summary(rows)
            except Exception as exc:
                self.logger.log(self.name, f"Conv {conv_id} : échec résumé LLM ({exc}), repli sur le résumé brut", "debug")
                summary = build_local_summary(rows)
                
            self.store.archive_messages_with_summary(rows, summary, conversation_id=conv_id)
            self.store.log_event("info", self.name, f"Conv {conv_id} : {len(rows)} messages résumés et archivés.")
            self.logger.log(self.name, f"Conv {conv_id} : {len(rows)} messages résumés et archivés", "debug")


class RetrievalAgent(BackgroundAgent):
    def __init__(self, store, stop_event, logger):
        super().__init__("retrieval-agent", store, stop_event, logger, interval=3)

    def tick(self):
        indexed = 0
        for row in self.store.get_unindexed_messages(limit=60):
            self.store.index_message_terms(row["id"], extract_terms(row["content"]))
            indexed += 1
        if indexed:
            self.logger.log(self.name, f"{indexed} message(s) indexé(s)", "all")
