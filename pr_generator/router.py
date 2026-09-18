from fastapi import APIRouter, HTTPException, Query, Response, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
from pr_generator.factory import JanDataFactory
import pandas as pd
import io
import json

router = APIRouter(tags=["Data Generator"])
factory = JanDataFactory()

# ─────────────────────────────────────────────
# Modèles de requêtes Pydantic
# ─────────────────────────────────────────────
class GenerateRequest(BaseModel):
    project: str
    topic: str
    label: str
    count: Optional[int] = 1
    template: Optional[str] = "default"
    variables: Optional[List[str]] = None

class UpdateDataRequest(BaseModel):
    text: str
    label: Optional[str] = ""

class AugmentRequest(BaseModel):
    variations: Optional[int] = 3

class SaveTemplateRequest(BaseModel):
    template: str

# ─────────────────────────────────────────────
# Endpoints FastAPI
# ─────────────────────────────────────────────

@router.post("/api/generate", status_code=202)
def api_generate(req: GenerateRequest):
    try:
        tid = factory.start_task(
            req.project,
            req.topic,
            req.label,
            req.count,
            req.template,
            req.variables
        )
        return {"task_id": tid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/task/{tid}")
def api_status(tid: str):
    session = factory.Session()
    try:
        task = session.query(factory.Task).filter_by(id=tid).first()
        if not task:
            # Fallback : recherche de la tâche dans la base SQLite du JanAgent
            try:
                from jan_agent.database import SQLiteStore
                sqlite_store = SQLiteStore("agent_memory.sqlite3")
                db_task = sqlite_store.get_task_result(int(tid))
                if db_task:
                    return db_task
            except Exception:
                pass
            raise HTTPException(status_code=404, detail="Tâche inconnue")
        return {
            "status": task.status,
            "progress": task.progress,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None
        }
    finally:
        session.close()

@router.get("/api/data/{project_name}")
def api_get_data(project_name: str):
    session = factory.Session()
    try:
        rows = session.query(factory.GeneratedData).join(factory.Project).filter(
            factory.Project.name == project_name
        ).all()
        import json
        res = []
        for d in rows:
            inputs_dict = None
            if d.inputs:
                try:
                    inputs_dict = json.loads(d.inputs)
                except Exception:
                    pass
            res.append({
                "id": d.id,
                "text": d.content,
                "label": d.label,
                "date": d.created_at,
                "inputs": inputs_dict
            })
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@router.get("/api/projects")
def api_list_projects():
    session = factory.Session()
    try:
        projects = session.query(factory.Project).all()
        import json
        res = []
        for p in projects:
            vars_list = None
            if p.variables:
                try:
                    vars_list = json.loads(p.variables)
                except Exception:
                    pass
            res.append({
                "id": p.id,
                "name": p.name,
                "variables": vars_list
            })
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@router.delete("/api/data/delete/{data_id}")
def api_delete_data(data_id: int):
    result, status = factory.delete_data(data_id)
    if status != 200:
        raise HTTPException(status_code=status, detail=result.get("error", "Erreur lors de la suppression"))
    return result

@router.put("/api/data/{data_id}")
def api_update_data(data_id: int, req: UpdateDataRequest):
    result, status = factory.update_data(data_id, req.text, req.label)
    if status != 200:
        raise HTTPException(status_code=status, detail=result.get("error", "Erreur lors de la mise à jour"))
    return result

@router.post("/api/data/augment/{data_id}")
def api_augment_data(data_id: int, req: Optional[AugmentRequest] = None):
    variations = req.variations if req else 3
    result, status = factory.augment_data(data_id, variations)
    if status not in (200, 201):
        raise HTTPException(status_code=status, detail=result.get("error", "Erreur lors de l'augmentation"))
    return result

@router.get("/api/export/{project_name}")
def api_export(project_name: str, format: str = "json"):
    content, mimetype = factory.export_data(project_name, format)
    if content is None:
        raise HTTPException(status_code=400, detail="Format non supporté")
    
    ext = "json" if format == "json" else "csv"
    return Response(
        content=content,
        media_type=mimetype,
        headers={"Content-Disposition": f"attachment; filename={project_name}_export.{ext}"}
    )

@router.get("/api/templates")
def api_list_templates():
    return {"templates": list(factory.template_manager.templates.keys())}

@router.get("/api/templates/{name}")
def api_get_template(name: str):
    template = factory.template_manager.get_template(name)
    return {"name": name, "template": template}

@router.post("/api/templates/{name}", status_code=201)
def api_save_template(name: str, req: SaveTemplateRequest):
    try:
        factory.template_manager.save_template(name, req.template)
        return {"message": f"Template '{name}' sauvegardé"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/import/{project_name}", status_code=201)
async def api_import_data(project_name: str, file: UploadFile = File(...)):
    filename = file.filename.lower()
    content = await file.read()
    
    try:
        if filename.endswith(".json"):
            data = json.loads(content.decode("utf-8"))
            if not isinstance(data, list):
                raise ValueError("Le fichier JSON doit contenir une liste d'objets.")
            df = pd.DataFrame(data)
        elif filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError("Format de fichier non supporté. Extensions acceptées : .json, .csv, .xls, .xlsx")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la lecture du fichier : {str(e)}")
        
    if df.empty:
        raise HTTPException(status_code=400, detail="Le fichier importé est vide.")
        
    # Identifier la colonne label
    cols = list(df.columns)
    label_col = None
    for col in cols:
        if str(col).lower() == "label":
            label_col = col
            break
    if not label_col:
        label_col = cols[-1] # par défaut, la dernière colonne
        
    input_cols = [c for c in cols if c != label_col]
    if not input_cols:
        raise HTTPException(status_code=400, detail="Aucune colonne d'entrée trouvée à part la colonne d'étiquette.")
        
    session = factory.Session()
    try:
        # Recherche ou création du projet
        project = session.query(factory.Project).filter_by(name=project_name).first()
        if not project:
            project = factory.Project(
                name=project_name,
                variables=json.dumps(input_cols, ensure_ascii=False)
            )
            session.add(project)
            session.commit()
            session.refresh(project)
        else:
            # Si le projet existe mais n'a pas encore de variables définies
            if not project.variables:
                project.variables = json.dumps(input_cols, ensure_ascii=False)
                session.commit()
                
        # Insertion des lignes
        for _, row in df.iterrows():
            row_label = str(row[label_col]) if pd.notna(row[label_col]) else ""
            row_inputs = {}
            for col in input_cols:
                val = row[col]
                row_inputs[str(col)] = str(val) if pd.notna(val) else ""
                
            # Représentation textuelle simplifiée
            if len(input_cols) == 1:
                content_text = row_inputs[input_cols[0]]
            else:
                content_text = " | ".join([f"{col}: {row_inputs[col]}" for col in input_cols])
                
            new_entry = factory.GeneratedData(
                project_id=project.id,
                content=content_text,
                label=row_label,
                inputs=json.dumps(row_inputs, ensure_ascii=False)
            )
            session.add(new_entry)
            
        session.commit()
        return {"message": f"{len(df)} lignes importées avec succès pour le projet '{project_name}'."}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Erreur d'insertion en base de données : {str(e)}")
    finally:
        session.close()
