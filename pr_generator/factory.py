import os
import json
import requests
import re
import sys
import threading
import uuid
import logging
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import QueuePool
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TemplateManager:
    """Gestionnaire de templates de prompts configurables."""
    
    DEFAULT_TEMPLATE = """Tu es un générateur de données. Génère EXACTEMENT {batch_size} entrées JSON.

RÈGLES:
- Retourne UN SEUL tableau JSON avec {batch_size} objets
- Chaque objet: {{"text": "...", "label": "{label}"}}
- Pas de texte hors JSON, pas de markdown

FORMAT: [{{"text": "...", "label": "{label}"}}, ...]"""

    def __init__(self, config_path: str = "templates.json"):
        if config_path == "templates.json":
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "templates.json")
        self.config_path = config_path
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"default": self.DEFAULT_TEMPLATE}
    
    def get_template(self, name: str = "default") -> str:
        return self.templates.get(name, self.DEFAULT_TEMPLATE)
    
    def save_template(self, name: str, template: str):
        self.templates[name] = template
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.templates, f, indent=2)


# ─────────────────────────────────────────────
# Base ORM
# ─────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


def make_models(schema: str | None):
    """
    Crée dynamiquement les classes ORM avec le bon schéma.
    """
    schema_kw = {"schema": schema} if schema else {}
    
    proj_args = {"extend_existing": True}
    proj_args.update(schema_kw)

    class Project(Base):
        __tablename__ = "projects"
        __table_args__ = proj_args
        id        = Column(Integer, primary_key=True)
        name      = Column(String(100), unique=True, nullable=False)
        variables = Column(Text, nullable=True) # JSON list of variable names

    fk_projects = f"{schema}.projects.id" if schema else "projects.id"

    data_args = {"extend_existing": True}
    data_args.update(schema_kw)

    class GeneratedData(Base):
        __tablename__ = "generated_data"
        __table_args__ = data_args
        id         = Column(Integer, primary_key=True)
        project_id = Column(Integer, ForeignKey(fk_projects, ondelete="CASCADE"))
        content    = Column(Text, nullable=False)
        label      = Column(String(50))
        created_at = Column(DateTime, default=datetime.utcnow)
        inputs     = Column(Text, nullable=True) # JSON object of input variables

    task_args = {"extend_existing": True}
    task_args.update(schema_kw)

    class Task(Base):
        __tablename__ = "tasks"
        __table_args__ = task_args
        id           = Column(String(50), primary_key=True)
        status       = Column(String(20))
        progress     = Column(Integer, default=0)
        project_name = Column(String(100))
        created_at   = Column(DateTime, default=datetime.utcnow)
        completed_at = Column(DateTime, nullable=True)

    return Project, GeneratedData, Task


# ─────────────────────────────────────────────
# Factory principale
# ─────────────────────────────────────────────
class JanDataFactory:
    def __init__(self, config_path: str = "data_config.json"):
        if config_path == "data_config.json":
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "data_config.json")

        if not os.path.exists(config_path):
            logger.error(f"Erreur : {config_path} introuvable.")
            sys.exit(1)

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        db = self.cfg["database"]
        self.schema = db.get("schema") or None

        self.db_url = (
            f"{db['driver']}://{db['user']}:{db['password']}"
            f"@{db['host']}:{db['port']}/{db['name']}"
        )

        connect_args = {}
        if self.schema:
            connect_args = {"options": f"-csearch_path={self.schema},public"}

        self.engine = create_engine(
            self.db_url,
            connect_args=connect_args,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True,
        )

        self.Project, self.GeneratedData, self.Task = make_models(self.schema)
        self.template_manager = TemplateManager()

        try:
            with self.engine.connect() as conn:
                if self.schema:
                    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema}"))
                    conn.commit()
            Base.metadata.create_all(self.engine)
            
            # Auto-migration: ajouter les nouvelles colonnes si elles n'existent pas
            from sqlalchemy import inspect
            inspector = inspect(self.engine)
            
            # Check projects table
            proj_cols = [c["name"] for c in inspector.get_columns("projects", schema=self.schema)]
            if "variables" not in proj_cols:
                logger.info("Ajout de la colonne 'variables' à la table 'projects'...")
                with self.engine.connect() as conn:
                    table_ref = f"{self.schema}.projects" if self.schema else "projects"
                    conn.execute(text(f"ALTER TABLE {table_ref} ADD COLUMN variables TEXT"))
                    conn.commit()
                    
            # Check generated_data table
            data_cols = [c["name"] for c in inspector.get_columns("generated_data", schema=self.schema)]
            if "inputs" not in data_cols:
                logger.info("Ajout de la colonne 'inputs' à la table 'generated_data'...")
                with self.engine.connect() as conn:
                    table_ref = f"{self.schema}.generated_data" if self.schema else "generated_data"
                    conn.execute(text(f"ALTER TABLE {table_ref} ADD COLUMN inputs TEXT"))
                    conn.commit()
            
            logger.info("Base de données initialisée et mise à niveau effectuée.")
        except Exception as exc:
            logger.error(f"ERREUR DB : {exc}")
            sys.exit(1)

        self.Session = sessionmaker(bind=self.engine)

        srv = self.cfg["server"]
        self.url     = srv["url"].rstrip("/")
        self.token   = srv["token"]
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type":  "application/json",
        }
        self.selected_model: str | None = None

        # Tenter d'auto-sélectionner le premier modèle disponible si aucun n'est spécifié
        resp = self.list_models()
        if resp and "data" in resp and resp["data"]:
            self.selected_model = resp["data"][0]["id"]
            logger.info(f"Modèle sélectionné automatiquement : {self.selected_model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry_error_callback=lambda retry_state: None
    )
    def _call_jan_api(self, payload: dict) -> dict | None:
        """Appel API Jan avec retry automatique."""
        try:
            r = requests.post(
                f"{self.url}/v1/chat/completions",
                json=payload,
                headers=self.headers,
                timeout=60,
            )
            if r.status_code == 200:
                return r.json()
            logger.warning(f"API Jan HTTP {r.status_code}")
            return None
        except Exception as e:
            logger.error(f"Erreur API Jan: {e}")
            raise

    def delete_data(self, data_id: int) -> Tuple[dict, int]:
        """Supprime une entrée spécifique par son ID."""
        session = self.Session()
        try:
            item = session.query(self.GeneratedData).filter_by(id=data_id).first()
            if not item:
                return {"error": "Donnée introuvable"}, 404
            
            session.delete(item)
            session.commit()
            logger.info(f"Donnée {data_id} supprimée")
            return {"message": "Donnée supprimée avec succès"}, 200
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur suppression {data_id}: {e}")
            return {"error": str(e)}, 500
        finally:
            session.close()

    def update_data(self, data_id: int, content: str, label: str) -> Tuple[dict, int]:
        """Met à jour une entrée existante."""
        session = self.Session()
        try:
            item = session.query(self.GeneratedData).filter_by(id=data_id).first()
            if not item:
                return {"error": "Donnée introuvable"}, 404
            
            item.content = content
            item.label = label
            session.commit()
            logger.info(f"Donnée {data_id} mise à jour")
            return {"message": "Donnée mise à jour", "id": data_id}, 200
        except Exception as e:
            session.rollback()
            logger.error(f"Erreur mise à jour {data_id}: {e}")
            return {"error": str(e)}, 500
        finally:
            session.close()

    def augment_data(self, data_id: int, variations: int = 3) -> Tuple[dict, int]:
        """Génère des variations paraphrasées d'une entrée existante."""
        session = self.Session()
        try:
            item = session.query(self.GeneratedData).filter_by(id=data_id).first()
            if not item:
                return {"error": "Donnée introuvable"}, 404

            # Utiliser le modèle sélectionné ou le premier dispo
            if not self.selected_model:
                resp = self.list_models()
                if resp and "data" in resp and resp["data"]:
                    self.selected_model = resp["data"][0]["id"]

            payload = {
                "model": self.selected_model,
                "messages": [
                    {
                        "role": "system",
                        "content": f"""Génère {variations} variations paraphrasées du texte suivant.
Retourne UN tableau JSON: [{{"text": "variation1"}}, {{"text": "variation2"}}, ...]
Label conservé: {item.label}"""
                    },
                    {
                        "role": "user",
                        "content": f"Texte original: {item.content}"
                    }
                ]
            }

            response = self._call_jan_api(payload)
            if not response:
                return {"error": "Échec génération variations"}, 500

            raw_content = response["choices"][0]["message"]["content"]
            variations_data = self._extract_all_json(raw_content)

            created_ids = []
            for var in variations_data:
                if isinstance(var, dict) and "text" in var:
                    new_entry = self.GeneratedData(
                        project_id=item.project_id,
                        content=var["text"],
                        label=item.label
                    )
                    session.add(new_entry)
                    session.flush()
                    created_ids.append(new_entry.id)
            
            session.commit()
            logger.info(f"{len(created_ids)} variations créées pour {data_id}")
            return {"message": f"{len(created_ids)} variations créées", "ids": created_ids}, 201

        except Exception as e:
            session.rollback()
            logger.error(f"Erreur augmentation {data_id}: {e}")
            return {"error": str(e)}, 500
        finally:
            session.close()

    def export_data(self, project_name: str, format: str = "json") -> Tuple[Any, str]:
        """Exporte les données dans différents formats."""
        session = self.Session()
        try:
            rows = session.query(self.GeneratedData).join(self.Project).filter(
                self.Project.name == project_name
            ).all()
            
            data = [{"id": r.id, "text": r.content, "label": r.label, "date": r.created_at.isoformat()} for r in rows]
            
            if format == "json":
                return json.dumps(data, ensure_ascii=False, indent=2), "application/json"
            elif format == "csv":
                import csv
                import io
                output = io.StringIO()
                if data:
                    writer = csv.DictWriter(output, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                return output.getvalue(), "text/csv"
            else:
                return None, ""
        finally:
            session.close()

    def list_models(self) -> dict | None:
        try:
            r = requests.get(f"{self.url}/v1/models", headers=self.headers, timeout=5)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    @staticmethod
    def _extract_all_json(raw: str) -> list[dict]:
        """
        Extrait tous les blocs JSON { ... } trouvés dans le texte brut.
        Gère le cas où le LLM renvoie plusieurs objets à la suite.
        """
        results = []
        blocks = re.findall(r"\{.*?\}", raw, re.DOTALL)
        
        for block in blocks:
            try:
                data = json.loads(block)
                if isinstance(data, dict):
                    results.append(data)
            except json.JSONDecodeError:
                continue
        return results

    # ── Worker en arrière-plan ────────────────────────────────────────────

    def _background_worker(
        self,
        task_id: str,
        project_name: str,
        prompt_topic: str,
        label: str,
        count: int,
        template_name: str = "default",
        variables: Optional[List[str]] = None,
    ) -> None:
        session = self.Session()
        try:
            project = session.query(self.Project).filter_by(name=project_name).first()
            if not project:
                project = self.Project(
                    name=project_name,
                    variables=json.dumps(variables, ensure_ascii=False) if variables else None
                )
                session.add(project)
                session.commit()
                session.refresh(project)
            else:
                if variables:
                    project.variables = json.dumps(variables, ensure_ascii=False)
                    session.commit()

            vars_list = json.loads(project.variables) if project.variables else None

            task = session.query(self.Task).filter_by(id=task_id).first()
            task.status = "En cours"
            session.commit()

            batch_size = 5
            remaining = count
            generated = 0
            
            template = self.template_manager.get_template(template_name)
            
            # S'assurer qu'un modèle est sélectionné
            if not self.selected_model:
                resp = self.list_models()
                if resp and "data" in resp and resp["data"]:
                    self.selected_model = resp["data"][0]["id"]

            while remaining > 0:
                current_batch = min(batch_size, remaining)
                
                # Construction du prompt système
                if vars_list:
                    keys_desc = ", ".join([f'"{v}"' for v in vars_list]) + ', "label"'
                    format_obj = {v: "..." for v in vars_list}
                    format_obj["label"] = label
                    format_json = json.dumps([format_obj], ensure_ascii=False)
                    
                    try:
                        system_content = template.format(
                            batch_size=current_batch,
                            label=label,
                            keys_desc=keys_desc,
                            format_json=format_json
                        )
                    except Exception:
                        system_content = f"""Tu es un générateur de données. Génère EXACTEMENT {current_batch} entrées JSON.

RÈGLES:
- Retourne UN SEUL tableau JSON avec {current_batch} objets
- Chaque objet doit contenir EXACTEMENT les clés suivantes : {keys_desc}
- Pas de texte hors JSON, pas de markdown

FORMAT DE RÉPONSE ATTENDU:
{format_json}"""
                else:
                    system_content = template.format(batch_size=current_batch, label=label)
                
                payload = {
                    "model": self.selected_model,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": f"Génère {current_batch} exemples sur: {prompt_topic}"},
                    ],
                }

                try:
                    response = self._call_jan_api(payload)
                    if response:
                        raw_content = response["choices"][0]["message"]["content"]
                        entries = self._extract_all_json(raw_content)
                        
                        if entries:
                            for entry in entries:
                                if isinstance(entry, dict):
                                    if vars_list:
                                        entry_inputs = {}
                                        for v in vars_list:
                                            entry_inputs[v] = entry.get(v, "")
                                        
                                        parts = [f"{v}: {entry_inputs[v]}" for v in vars_list]
                                        content_text = " | ".join(parts)
                                        inputs_val = json.dumps(entry_inputs, ensure_ascii=False)
                                    else:
                                        content_text = entry.get("text", entry.get("content", "Contenu vide"))
                                        inputs_val = None
                                        
                                    new_entry = self.GeneratedData(
                                        project_id=project.id,
                                        content=content_text,
                                        label=str(entry.get("label", label)),
                                        inputs=inputs_val
                                    )
                                    session.add(new_entry)
                                    generated += 1
                            session.commit()
                            logger.info(f"[{task_id}] Batch: {len(entries)} entrées")
                        else:
                            logger.warning(f"[{task_id}] Aucun JSON valide")
                    else:
                        logger.warning(f"[{task_id}] Échec API après retry")

                except Exception as e:
                    logger.error(f"[{task_id}] Erreur batch: {e}")

                remaining -= current_batch
                task.progress = int((generated / count) * 100)
                session.commit()

            task.status = "Terminé"
            task.completed_at = datetime.utcnow()
            session.commit()
            logger.info(f"[{task_id}] Terminé: {generated}/{count} entrées")

        except Exception as exc:
            logger.error(f"[{task_id}] Erreur critique: {exc}")
            try:
                session.rollback()
                t = session.query(self.Task).filter_by(id=task_id).first()
                if t:
                    t.status = "Erreur"
                    session.commit()
            except: pass
        finally:
            session.close()

    def start_task(
        self,
        project: str,
        topic: str,
        label: str,
        count: int,
        template: str = "default",
        variables: Optional[List[str]] = None
    ) -> str:
        session = self.Session()
        task_id = str(uuid.uuid4())[:8]
        session.add(self.Task(id=task_id, status="En attente", progress=0, project_name=project))
        session.commit()
        session.close()

        threading.Thread(
            target=self._background_worker,
            args=(task_id, project, topic, label, count, template, variables),
            daemon=True,
        ).start()
        return task_id
