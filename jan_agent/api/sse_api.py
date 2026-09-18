"""
API SSE (Server-Sent Events) pour JanAgent.

Remplace le WebSocket par une communication unidirectionnelle plus simple :
- POST /api/v2/chat/send          → déclenche une tâche agent (renvoie un task_id)
- GET  /api/v2/chat/stream/{id}   → stream SSE des événements de la tâche
- POST /api/v2/chat/approve/{id}  → répond à une demande d'approbation en cours
- GET  /api/v2/chat/status/{id}   → statut rapide (non-streaming)
- POST /api/v2/chat/cancel/{id}   → annule une tâche en cours

Les autres endpoints REST (historique, sécurité, tâches) restent sur /api/...
"""

import asyncio
import json
import queue
import threading
import uuid
from typing import Optional

from fastapi import HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from jan_agent.core import JanAgent
from jan_agent.utils import resolve_model_id


# ─────────────────────────────────────────────
# Modèles Pydantic
# ─────────────────────────────────────────────

class SendMessageRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None
    agent_mode: Optional[str] = "loop" # "loop" | "generative"
    model_id: Optional[str] = None

class ApproveRequest(BaseModel):
    choice: str  # "y" | "n" | "a"


# ─────────────────────────────────────────────
# Gestionnaire de sessions
# ─────────────────────────────────────────────

class AgentSession:
    """Représente une exécution de l'agent en cours."""

    def __init__(self, task_id: str, conversation_id: int, store, agent_mode: str = "loop", model_id: Optional[str] = None):
        self.task_id = task_id
        self.conversation_id = conversation_id
        self.agent_mode = agent_mode
        self.model_id = model_id
        self.store = store
        self.events: queue.Queue = queue.Queue()     # file d'événements SSE
        self.approval: queue.Queue = queue.Queue(maxsize=1)  # réponse d'approbation
        self._status = "pending"     # pending | running | waiting_approval | done | error | cancelled
        self.done = False
        self._thread: Optional[threading.Thread] = None
        try:
            self.store.create_chat_session(task_id, conversation_id, agent_mode, model_id, status=self._status)
        except Exception:
            pass

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value
        try:
            self.store.update_chat_session_status(self.task_id, value)
        except Exception:
            pass

    # ── Helpers pour émettre des événements ──────────────────────────────────

    def emit(self, event_type: str, content):
        """Ajoute un événement SSE dans la file."""
        self.events.put({"type": "event", "event": event_type, "content": content})

    def emit_approval_needed(self, tool_name: str, args: dict):
        """Signale qu'une approbation est requise."""
        self.status = "waiting_approval"
        self.events.put({"type": "action_required", "tool": tool_name, "args": args})

    def emit_done(self):
        """Signale la fin de l'exécution."""
        self.status = "done"
        self.done = True
        self.events.put({"type": "done"})

    def emit_error(self, message: str):
        """Signale une erreur."""
        self.status = "error"
        self.done = True
        self.events.put({"type": "error", "content": message})

    # ── Callbacks pour l'agent ───────────────────────────────────────────────

    def output_callback(self, event_type: str, content):
        """Appelé par l'agent pour envoyer un événement."""
        if event_type in ("thought", "thought_delta", "tool_requested", "tool_executed",
                          "observation_received", "response", "response_delta", "security_error", "info"):
            self.emit(event_type, content)

    def input_callback(self, tool_name: str, args: dict) -> str:
        """Appelé par l'agent pour demander une approbation (BLOQUANT côté thread agent)."""
        self.emit_approval_needed(tool_name, args)
        # Attend la réponse côté client (timeout 5 minutes)
        try:
            choice = self.approval.get(timeout=300)
        except queue.Empty:
            choice = "n"
        self.status = "running"
        return choice


class SessionManager:
    """Gère toutes les sessions actives."""

    def __init__(self, agent: JanAgent):
        self._agent = agent
        self._sessions: dict[str, AgentSession] = {}
        self._lock = threading.Lock()
        self._agent_lock = threading.Lock()   # une seule exécution à la fois
        # Les sessions vivent uniquement en mémoire : toute session encore "active"
        # en base au démarrage provient forcément d'un arrêt précédent du serveur.
        try:
            self._agent.store.mark_stale_chat_sessions_interrupted()
        except Exception:
            pass

    def create_session(self, conversation_id: int, agent_mode: str = "loop", model_id: Optional[str] = None) -> AgentSession:
        task_id = str(uuid.uuid4())
        session = AgentSession(task_id, conversation_id, self._agent.store, agent_mode=agent_mode, model_id=model_id)
        with self._lock:
            self._sessions[task_id] = session
        return session

    def get_session(self, task_id: str) -> Optional[AgentSession]:
        with self._lock:
            return self._sessions.get(task_id)

    def remove_session(self, task_id: str):
        with self._lock:
            self._sessions.pop(task_id, None)

    def run_agent_in_background(self, session: AgentSession, message: str):
        """Lance l'exécution de l'agent dans un thread séparé."""

        def _run():
            if not self._agent_lock.acquire(timeout=2):
                session.emit_error("L'agent est déjà occupé sur une autre tâche.")
                return

            try:
                session.status = "running"

                try:
                    if session.model_id:
                        model_id = session.model_id
                    else:
                        default_model = self._agent.cfg.get("model", {}).get("default", "default")
                        try:
                            model_id = resolve_model_id(
                                self._agent.url, self._agent.headers, default_model, timeout=5
                            )
                        except Exception:
                            model_id = default_model

                    self._agent.generate_completion(
                        model_id, message,
                        conversation_id=session.conversation_id,
                        agent_mode=session.agent_mode,
                        output_callback=session.output_callback,
                        input_callback=session.input_callback,
                    )

                except Exception as exc:
                    session.emit_error(f"Erreur d'exécution : {str(exc)}")
                finally:
                    session.emit_done()
            finally:
                self._agent_lock.release()

        t = threading.Thread(target=_run, daemon=True)
        session._thread = t
        t.start()


# ─────────────────────────────────────────────
# Constructeur de l'API SSE
# ─────────────────────────────────────────────

def build_sse_router(agent: JanAgent, verify_jwt):
    """
    Construit et retourne un APIRouter FastAPI avec tous les endpoints SSE.
    À monter sur l'app principale avec prefix="/api/v2".
    """
    from fastapi import APIRouter
    router = APIRouter()
    manager = SessionManager(agent)

    # ── GET /chat/models ─────────────────────────────────────────────────────

    @router.get("/chat/models", dependencies=[Depends(verify_jwt)])
    def list_chat_models():
        """Récupère la liste des modèles disponibles auprès du serveur Jan."""
        models_data = agent.list_models()
        if models_data is None:
            raise HTTPException(status_code=502, detail="Impossible de récupérer les modèles depuis le serveur Jan.")
        return models_data

    # ── POST /chat/send ──────────────────────────────────────────────────────

    @router.post("/chat/send", dependencies=[Depends(verify_jwt)])
    def send_message(req: SendMessageRequest, username: str = Depends(verify_jwt)):
        """Démarre une session agent et retourne un task_id."""
        if not req.message.strip():
            raise HTTPException(status_code=400, detail="Le message ne peut pas être vide.")

        conversation_id = req.conversation_id
        if not conversation_id:
            # Create a new conversation
            title = req.message[:30] + "..." if len(req.message) > 30 else req.message
            conversation_id = agent.store.create_conversation(username, title)

        session = manager.create_session(conversation_id, agent_mode=req.agent_mode, model_id=req.model_id)
        manager.run_agent_in_background(session, req.message.strip())

        return {
            "task_id": session.task_id,
            "conversation_id": conversation_id,
            "stream_url": f"/api/v2/chat/stream/{session.task_id}",
            "status": "started"
        }

    # ── GET /chat/stream/{task_id} ───────────────────────────────────────────

    @router.get("/chat/stream/{task_id}")
    def stream_events(task_id: str):
        """
        Ouvre un flux SSE qui pousse les événements de la session en temps réel.
        Chaque événement est au format :  data: <json>\\n\\n
        """
        session = manager.get_session(task_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session introuvable.")

        def _event_generator():
            # Ping initial pour établir la connexion SSE
            yield "data: " + json.dumps({"type": "connected", "task_id": task_id}) + "\n\n"

            while True:
                try:
                    event = session.events.get(timeout=30)
                    yield "data: " + json.dumps(event) + "\n\n"

                    if event.get("type") in ("done", "error"):
                        # Nettoyage différé (5s) pour laisser le client traiter
                        def _cleanup():
                            import time
                            time.sleep(5)
                            manager.remove_session(task_id)
                        threading.Thread(target=_cleanup, daemon=True).start()
                        break

                except queue.Empty:
                    # Heartbeat pour maintenir la connexion ouverte
                    yield ": heartbeat\n\n"

                    if session.done:
                        break

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── POST /chat/approve/{task_id} ─────────────────────────────────────────

    @router.post("/chat/approve/{task_id}")
    def approve_action(task_id: str, req: ApproveRequest):
        """Répond à une demande d'approbation en attente."""
        session = manager.get_session(task_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session introuvable.")

        choice = req.choice.strip().lower()
        if choice not in ("y", "n", "a"):
            raise HTTPException(status_code=400, detail="Choix invalide. Utilisez 'y', 'n', ou 'a'.")

        if session.status != "waiting_approval":
            raise HTTPException(status_code=409, detail="Aucune approbation en attente pour cette session.")

        try:
            session.approval.put_nowait(choice)
        except queue.Full:
            raise HTTPException(status_code=409, detail="Une réponse a déjà été envoyée.")

        return {"status": "approved", "choice": choice}

    # ── GET /chat/status/{task_id} ───────────────────────────────────────────

    @router.get("/chat/status/{task_id}")
    def get_status(task_id: str):
        """
        Retourne le statut actuel d'une session (non-streaming).
        Si la session n'est plus en mémoire (terminée, ou serveur redémarré),
        retombe sur le dernier statut connu en base.
        """
        session = manager.get_session(task_id)
        if session is not None:
            return {"task_id": task_id, "status": session.status}

        db_session = agent.store.get_chat_session(task_id)
        if db_session is None:
            return {"task_id": task_id, "status": "not_found"}
        return {"task_id": task_id, "status": db_session["status"]}

    # ── POST /chat/cancel/{task_id} ──────────────────────────────────────────

    @router.post("/chat/cancel/{task_id}")
    def cancel_session(task_id: str):
        """Annule une session en cours en envoyant 'n' à toute approbation pendante."""
        session = manager.get_session(task_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session introuvable.")

        session.status = "cancelled"
        session.done = True

        # Débloquer le thread agent s'il attend une approbation
        try:
            session.approval.put_nowait("n")
        except queue.Full:
            pass

        session.emit_error("Session annulée par l'utilisateur.")
        manager.remove_session(task_id)
        return {"status": "cancelled", "task_id": task_id}

    return router
