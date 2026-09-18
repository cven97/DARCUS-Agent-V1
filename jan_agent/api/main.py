import os
import asyncio
import json
import queue
import threading
import jwt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from jan_agent.core import JanAgent
from jan_agent.utils import resolve_model_id
from jan_agent.api.schemas import ChatRequest, AutoApproveRequest, ModelConfigUpdate
from jan_agent.api.sse_api import build_sse_router
from pr_generator.router import router as generator_router

app = FastAPI(
    title="JanAgent API",
    description="API REST + SSE pour l'agent autonome JanAgent",
    version="2.0.0",
)

# Configuration CORS permissive pour l'intégration frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JWT_SECRET = os.environ.get("JWT_SECRET", "change-me-in-production-use-a-long-random-string-of-32+chars")
JWT_ALGORITHM = "HS256"
security = HTTPBearer()

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("sub", "")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token a expiré")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token invalide")

# Initialisation globale de l'agent
agent = JanAgent(config_path="config.json")
agent_execution_lock = threading.Lock()

# ── Monter le router SSE v2 ──────────────────────────────────────────────────
sse_router = build_sse_router(agent, verify_jwt)
app.include_router(sse_router, prefix="/api/v2", tags=["SSE v2"])

# ── Monter le router Generator ───────────────────────────────────────────────
app.include_router(generator_router)

# ── Monter le router ML Models ────────────────────────────────────────────────
from ml_models.router import router as ml_router
app.include_router(ml_router)

@app.get("/")
def read_root():
    return {"status": "online", "message": "JanAgent API is running."}

from pydantic import BaseModel

class ConversationCreate(BaseModel):
    title: str = "Nouvelle Conversation"

@app.get("/api/conversations", dependencies=[Depends(verify_jwt)])
def get_conversations(username: str = Depends(verify_jwt)):
    try:
        return agent.store.get_conversations(username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversations", dependencies=[Depends(verify_jwt)])
def create_conversation(req: ConversationCreate, username: str = Depends(verify_jwt)):
    try:
        conv_id = agent.store.create_conversation(username, req.title)
        return {"id": conv_id, "title": req.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversations/{conv_id}", dependencies=[Depends(verify_jwt)])
def delete_conversation(conv_id: int, username: str = Depends(verify_jwt)):
    try:
        agent.store.delete_conversation(conv_id, username)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/{conversation_id}/history", dependencies=[Depends(verify_jwt)])
def get_chat_history(conversation_id: int):
    """
    Récupère l'historique complet des messages depuis SQLite pour une conversation.
    """
    try:
        with agent.store.lock:
            rows = agent.store.conn.execute(
                "SELECT id, role, content, created_at, archived FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,)
            ).fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération de l'historique : {str(e)}")

@app.post("/api/chat/{conversation_id}/reset", dependencies=[Depends(verify_jwt)])
def reset_chat_history(conversation_id: int):
    """
    Réinitialise la mémoire conversationnelle d'une conversation.
    """
    try:
        agent.reset_memory(conversation_id)
        return {"status": "success", "message": "Conversation history reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la réinitialisation : {str(e)}")

@app.get("/api/chat/{conversation_id}/plan", dependencies=[Depends(verify_jwt)])
def get_active_plan(conversation_id: int):
    """
    Récupère le plan d'action actif pour la conversation donnée.
    """
    try:
        plan = agent.store.get_active_plan(conversation_id)
        return plan if plan else {"status": "no_active_plan"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks", dependencies=[Depends(verify_jwt)])
def get_tasks():
    """
    Liste les tâches d'arrière-plan enregistrées.
    """
    try:
        with agent.store.lock:
            rows = agent.store.conn.execute(
                "SELECT id, kind, payload, status, result, error, created_at, updated_at FROM tasks ORDER BY id DESC"
            ).fetchall()
        
        tasks_list = []
        for row in rows:
            d = dict(row)
            try:
                d["payload"] = json.loads(d["payload"])
            except Exception:
                pass
            try:
                if d["result"]:
                    d["result"] = json.loads(d["result"])
            except Exception:
                pass
            tasks_list.append(d)
        return tasks_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des tâches : {str(e)}")

@app.get("/api/config", dependencies=[Depends(verify_jwt)])
def get_model_config():
    """
    Récupère la configuration actuelle du modèle.
    """
    try:
        return {
            "server_url": agent.cfg.get("server", {}).get("url", ""),
            "default_model": agent.cfg.get("model", {}).get("default", ""),
            "temperature": agent.cfg.get("model", {}).get("temperature", 0.3),
            "max_tokens": agent.cfg.get("model", {}).get("max_tokens", 2048),
            "system_prompt": agent.cfg.get("system_prompt", ""),
            "memory_prompt": agent.cfg.get("memory_prompt", "Fais un résumé logique condensé des échanges suivants. Conserve uniquement les décisions prises, les chemins de fichiers mentionnés et les résultats des tâches d'exécution. Sois concis."),
            "planner_prompt": agent.cfg.get("planner_prompt", "Tu es un planificateur expert. L'utilisateur a une requête. Ta tâche est de décomposer cette requête en un plan d'action séquentiel.\nRÈGLES IMPORTANTES :\n1. Ne réponds pas à la question de l'utilisateur.\n2. Fournis UNIQUEMENT un tableau JSON de chaînes de caractères.\n3. Le nombre d'étapes DOIT être proportionnel à la complexité. Pour une requête simple, génère EXACTEMENT 1 étape.\n4. N'ajoute AUCUN préfixe comme 'Étape 1 :' dans tes descriptions.")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération de la configuration: {str(e)}")

@app.post("/api/config", dependencies=[Depends(verify_jwt)])
def update_model_config(req: ModelConfigUpdate):
    """
    Met à jour la configuration du modèle.
    """
    try:
        agent.cfg.setdefault("server", {})["url"] = req.server_url
        agent.cfg.setdefault("model", {})["default"] = req.default_model
        agent.cfg.setdefault("model", {})["temperature"] = req.temperature
        agent.cfg.setdefault("model", {})["max_tokens"] = req.max_tokens
        if req.system_prompt is not None:
            agent.cfg["system_prompt"] = req.system_prompt
        if req.memory_prompt is not None:
            agent.cfg["memory_prompt"] = req.memory_prompt
        if req.planner_prompt is not None:
            agent.cfg["planner_prompt"] = req.planner_prompt
        
        # Sauvegarder sur disque, le ConfigAgent va détecter la modification et l'appliquer dynamiquement
        import json
        agent.config_path.write_text(json.dumps(agent.cfg, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
        
        return {"status": "success", "message": "Configuration mise à jour avec succès"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la mise à jour de la configuration: {str(e)}")

@app.get("/api/agent/status", dependencies=[Depends(verify_jwt)])
def get_agent_status():
    """
    Récupère l'état de supervision des agents, métriques et logs système.
    """
    try:
        agents_list = []
        if hasattr(agent, "runtime") and hasattr(agent.runtime, "agents"):
            for a in agent.runtime.agents:
                agents_list.append({
                    "name": a.name,
                    "is_alive": a.is_alive(),
                    "interval": getattr(a, "interval", None),
                    "type": "background"
                })
        
        try:
            db_agents = agent.store.get_agents()
            for db_a in db_agents:
                agents_list.append({
                    "name": db_a["name"],
                    "is_alive": db_a["status"] == "running",
                    "interval": db_a["interval"],
                    "type": "specialist",
                    "role": db_a["role"],
                    "last_seen": db_a["last_seen"]
                })
        except Exception as e:
            pass
        
        metrics_dict = {}
        with agent.store.lock:
            metrics_rows = agent.store.conn.execute("""
                SELECT name, value, created_at FROM metrics m1
                WHERE id = (SELECT MAX(id) FROM metrics m2 WHERE m2.name = m1.name)
            """).fetchall()
            for r in metrics_rows:
                metrics_dict[r["name"]] = r["value"]
                
        with agent.store.lock:
            event_rows = agent.store.conn.execute("""
                SELECT id, level, source, message, created_at 
                FROM events 
                ORDER BY id DESC 
                LIMIT 50
            """).fetchall()
        events_list = [dict(r) for r in event_rows]
        
        return {
            "agents": agents_list,
            "metrics": metrics_dict,
            "events": events_list,
            "system_prompt": agent.cfg.get("system_prompt", ""),
            "memory_prompt": agent.cfg.get("memory_prompt", "Fais un résumé logique condensé des échanges suivants. Conserve uniquement les décisions prises, les chemins de fichiers mentionnés et les résultats des tâches d'exécution. Sois concis."),
            "planner_prompt": agent.cfg.get("planner_prompt", "Tu es un planificateur expert. L'utilisateur a une requête. Ta tâche est de décomposer cette requête en un plan d'action séquentiel.\nRÈGLES IMPORTANTES :\n1. Ne réponds pas à la question de l'utilisateur.\n2. Fournis UNIQUEMENT un tableau JSON de chaînes de caractères.\n3. Le nombre d'étapes DOIT être proportionnel à la complexité. Pour une requête simple, génère EXACTEMENT 1 étape.\n4. N'ajoute AUCUN préfixe comme 'Étape 1 :' dans tes descriptions.")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération du statut : {str(e)}")

@app.get("/api/security/config", dependencies=[Depends(verify_jwt)])
def get_security_config():
    """
    Récupère la configuration actuelle des règles de sécurité (whitelist, blacklist, auto-approbation).
    """
    try:
        policy = agent.store.get_security_policy()
        auto_approved = agent.cfg.get("security", {}).get("auto_approved_commands", [])
        return {
            "allowed_commands": list(policy.get("allowed_commands", [])),
            "forbidden_keywords": list(policy.get("forbidden_keywords", [])),
            "max_retry_count": policy.get("max_retry_count", 1),
            "auto_approved_commands": auto_approved
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération de la sécurité : {str(e)}")

@app.post("/api/security/auto-approve", dependencies=[Depends(verify_jwt)])
def add_auto_approve(req: AutoApproveRequest):
    """
    Ajoute une commande de base à la whitelist d'approbation automatique.
    """
    try:
        cmd_base = req.command_base.strip().lower()
        if not cmd_base:
            raise HTTPException(status_code=400, detail="Le nom de la commande ne peut pas être vide.")
        agent._add_to_auto_approved(cmd_base)
        return {"status": "success", "message": f"La commande '{cmd_base}' a été ajoutée à la whitelist d'approbation automatique."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'ajout à la whitelist : {str(e)}")

@app.websocket("/api/ws/agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    validation_queue = queue.Queue(maxsize=1)
    loop = asyncio.get_running_loop()
    
    def ws_output_callback(event_type, content):
        payload = {
            "type": "event",
            "event": event_type,
            "content": content
        }
        asyncio.run_coroutine_threadsafe(websocket.send_json(payload), loop)
        
    def ws_input_callback(tool_name, args):
        payload = {
            "type": "action_required",
            "tool": tool_name,
            "args": args
        }
        asyncio.run_coroutine_threadsafe(websocket.send_json(payload), loop)
        # Blocage en attente de la réponse du client WebSocket
        choice = validation_queue.get()
        return choice

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "chat":
                message = data.get("message")
                if not message:
                    await websocket.send_json({"type": "error", "content": "Le message est vide."})
                    continue
                
                if agent_execution_lock.locked():
                    await websocket.send_json({"type": "error", "content": "L'agent est déjà en cours d'exécution d'une tâche."})
                    continue
                
                agent_mode = data.get("agent_mode", "loop")
                def run_agent():
                    with agent_execution_lock:
                        try:
                            default_cfg_model = agent.cfg.get("model", {}).get("default", "default")
                            try:
                                active_model_id = resolve_model_id(agent.url, agent.headers, default_cfg_model, timeout=5)
                            except Exception:
                                active_model_id = default_cfg_model

                            agent.generate_completion(
                                active_model_id, message, agent_mode=agent_mode,
                                output_callback=ws_output_callback,
                                input_callback=ws_input_callback,
                            )
                        except Exception as e:
                            asyncio.run_coroutine_threadsafe(
                                websocket.send_json({"type": "error", "content": f"Erreur d'exécution : {str(e)}"}),
                                loop
                            )
                        finally:
                            asyncio.run_coroutine_threadsafe(
                                websocket.send_json({"type": "done"}),
                                loop
                            )
                
                # Exécution asynchrone dans un thread séparé pour ne pas bloquer le serveur web
                threading.Thread(target=run_agent, daemon=True).start()
                
            elif action == "approve":
                choice = data.get("choice", "n")
                if not validation_queue.empty():
                    try:
                        validation_queue.get_nowait()
                    except queue.Empty:
                        pass
                validation_queue.put_nowait(choice)
                
    except WebSocketDisconnect:
        # En cas de déconnexion, annuler le prompt bloquant de l'agent s'il y en a un
        if not validation_queue.empty():
            try:
                validation_queue.get_nowait()
            except queue.Empty:
                pass
        validation_queue.put_nowait("n")
