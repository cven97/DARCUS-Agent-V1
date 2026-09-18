import json
import sqlite3
import threading
from pathlib import Path
from collections import Counter
from .utils import now_ts, extract_terms

class SQLiteStore:
    def __init__(self, db_path="agent_memory.sqlite3"):
        self.path = Path(db_path)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        with self.lock, self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA busy_timeout=5000")
            
            try:
                self.conn.execute("ALTER TABLE messages ADD COLUMN conversation_id INTEGER NOT NULL DEFAULT 1")
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute("ALTER TABLE summaries ADD COLUMN conversation_id INTEGER NOT NULL DEFAULT 1")
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute("ALTER TABLE tasks ADD COLUMN target_agent TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute("ALTER TABLE tasks ADD COLUMN parent_task_id INTEGER")
            except sqlite3.OperationalError:
                pass

            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'stopped',
                    interval REAL,
                    last_seen TEXT
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL DEFAULT 1,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0,
                    indexed INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL DEFAULT 1,
                    content TEXT NOT NULL,
                    message_start_id INTEGER,
                    message_end_id INTEGER,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_index (
                    term TEXT NOT NULL,
                    message_id INTEGER NOT NULL,
                    weight INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (term, message_id)
                );
                CREATE TABLE IF NOT EXISTS security_rules (
                    kind TEXT NOT NULL,
                    value TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (kind, value)
                );
                CREATE TABLE IF NOT EXISTS plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS plan_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id INTEGER NOT NULL,
                    step_number INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    result_summary TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(plan_id) REFERENCES plans(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS ml_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    algorithm TEXT NOT NULL,
                    hyperparams TEXT,
                    status TEXT NOT NULL DEFAULT 'draft',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ml_model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    version_tag TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    dataset_path TEXT,
                    dataset_size INTEGER,
                    metrics TEXT,
                    training_duration_ms REAL,
                    is_active INTEGER DEFAULT 0,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(model_id) REFERENCES ml_models(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS ml_training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    version_id INTEGER,
                    action TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(model_id) REFERENCES ml_models(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    task_id TEXT PRIMARY KEY,
                    conversation_id INTEGER NOT NULL,
                    agent_mode TEXT NOT NULL DEFAULT 'loop',
                    model_id TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def close(self):
        with self.lock:
            self.conn.close()

    def log_event(self, level, source, message):
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO events(level, source, message, created_at) VALUES (?, ?, ?, ?)",
                (level, source, message, now_ts()),
            )

    def create_conversation(self, username, title="Nouvelle Conversation"):
        with self.lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO conversations(username, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (username, title, now_ts(), now_ts())
            )
            return cur.lastrowid

    def get_conversations(self, username):
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE username = ? ORDER BY updated_at DESC",
                (username,)
            ).fetchall()
        return [dict(row) for row in rows]
        
    def delete_conversation(self, conversation_id, username):
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            self.conn.execute("DELETE FROM summaries WHERE conversation_id = ?", (conversation_id,))
            self.conn.execute("DELETE FROM conversations WHERE id = ? AND username = ?", (conversation_id, username))

    def update_conversation_title(self, conversation_id, title):
        with self.lock, self.conn:
            self.conn.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (title, now_ts(), conversation_id))

    def add_message(self, role, content, conversation_id=1):
        with self.lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO messages(conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, now_ts()),
            )
            self.conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now_ts(), conversation_id))
            return cur.lastrowid

    def get_recent_messages(self, limit, conversation_id=1):
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT role, content FROM messages
                WHERE archived = 0 AND conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def count_active_messages(self, conversation_id=1):
        with self.lock:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE archived = 0 AND conversation_id = ?", (conversation_id,)
            ).fetchone()
        return row["n"]

    def get_conversations_needing_summary(self, threshold):
        with self.lock:
            rows = self.conn.execute(
                "SELECT conversation_id, COUNT(*) AS n FROM messages WHERE archived = 0 GROUP BY conversation_id HAVING n >= ?", (threshold,)
            ).fetchall()
        return [row["conversation_id"] for row in rows]

    def get_messages_to_summarize(self, keep_last, conversation_id=1):
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT id, role, content FROM messages
                WHERE archived = 0 AND conversation_id = ?
                ORDER BY id DESC
                LIMIT -1 OFFSET ?
                """,
                (conversation_id, keep_last),
            ).fetchall()
        return list(reversed(rows))

    def archive_messages_with_summary(self, rows, summary, conversation_id=1):
        if not rows:
            return
        ids = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in ids)
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO summaries(conversation_id, content, message_start_id, message_end_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, summary, ids[0], ids[-1], now_ts()),
            )
            self.conn.execute(
                f"UPDATE messages SET archived = 1 WHERE id IN ({placeholders})",
                ids,
            )

    def get_latest_summary(self, conversation_id=1):
        with self.lock:
            row = self.conn.execute(
                "SELECT content FROM summaries WHERE conversation_id = ? ORDER BY id DESC LIMIT 1", (conversation_id,)
            ).fetchone()
        return row["content"] if row else None

    def reset_conversation(self, conversation_id=1):
        with self.lock, self.conn:
            # For backward compatibility or specific reset
            self.conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            self.conn.execute("DELETE FROM summaries WHERE conversation_id = ?", (conversation_id,))

    def create_task(self, kind, payload, status="pending"):
        payload_json = json.dumps(payload, ensure_ascii=False)
        with self.lock, self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO tasks(kind, payload, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (kind, payload_json, status, now_ts(), now_ts()),
            )
            return cur.lastrowid

    def set_task_status(self, task_id, status):
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                (status, now_ts(), task_id),
            )

    def fetch_pending_task(self):
        with self.lock, self.conn:
            row = self.conn.execute(
                """
                SELECT id, kind, payload FROM tasks
                WHERE status = 'pending'
                ORDER BY id
                LIMIT 1
                """
            ).fetchone()
            if not row:
                return None
            self.conn.execute(
                "UPDATE tasks SET status = 'running', updated_at = ? WHERE id = ?",
                (now_ts(), row["id"]),
            )
        return row

    def finish_task(self, task_id, result=None, error=None):
        status = "done" if error is None else "failed"
        with self.lock, self.conn:
            self.conn.execute(
                """
                UPDATE tasks
                SET status = ?, result = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, result, error, now_ts(), task_id),
            )

    def get_task_result(self, task_id):
        with self.lock:
            row = self.conn.execute(
                "SELECT status, result, error FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        return dict(row) if row else None

    def put_setting(self, key, value):
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO settings(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, json.dumps(value, ensure_ascii=False), now_ts()),
            )

    def get_setting(self, key, default=None):
        with self.lock:
            row = self.conn.execute(
                "SELECT value FROM settings WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    def sync_security(self, security_cfg):
        allowed = sorted({str(cmd).lower().strip() for cmd in security_cfg.get("allowed_commands", []) if cmd})
        forbidden = sorted({str(kw).lower().strip() for kw in security_cfg.get("forbidden_keywords", []) if kw})
        max_retry = int(security_cfg.get("max_retry_count", 1))
        max_hallucination = int(security_cfg.get("max_hallucination_text_length", 10))

        with self.lock, self.conn:
            self.conn.execute("DELETE FROM security_rules")
            self.conn.executemany(
                """
                INSERT INTO security_rules(kind, value, enabled, updated_at)
                VALUES ('allowed_command', ?, 1, ?)
                """,
                [(cmd, now_ts()) for cmd in allowed],
            )
            self.conn.executemany(
                """
                INSERT INTO security_rules(kind, value, enabled, updated_at)
                VALUES ('forbidden_keyword', ?, 1, ?)
                """,
                [(keyword, now_ts()) for keyword in forbidden],
            )

        self.put_setting(
            "security_policy",
            {
                "allowed_commands": allowed,
                "forbidden_keywords": forbidden,
                "max_retry_count": max_retry,
                "max_hallucination_text_length": max_hallucination,
            },
        )

    def get_security_policy(self):
        policy = self.get_setting("security_policy", {})
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT kind, value FROM security_rules
                WHERE enabled = 1
                ORDER BY kind, value
                """
            ).fetchall()

        allowed = {row["value"] for row in rows if row["kind"] == "allowed_command"}
        forbidden = [row["value"] for row in rows if row["kind"] == "forbidden_keyword"]
        return {
            "allowed_commands": allowed,
            "forbidden_keywords": forbidden,
            "max_retry_count": int(policy.get("max_retry_count", 1)),
            "max_hallucination_text_length": int(policy.get("max_hallucination_text_length", 10)),
        }

    def put_metric(self, name, value):
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO metrics(name, value, created_at) VALUES (?, ?, ?)",
                (name, float(value), now_ts()),
            )

    def get_unindexed_messages(self, limit=50):
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT id, content FROM messages
                WHERE indexed = 0
                ORDER BY id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return rows

    def index_message_terms(self, message_id, terms):
        counts = Counter(terms)
        with self.lock, self.conn:
            for term, weight in counts.items():
                self.conn.execute(
                    """
                    INSERT INTO memory_index(term, message_id, weight)
                    VALUES (?, ?, ?)
                    ON CONFLICT(term, message_id) DO UPDATE SET weight = excluded.weight
                    """,
                    (term, message_id, weight),
                )
            self.conn.execute("UPDATE messages SET indexed = 1 WHERE id = ?", (message_id,))

    def retrieve_context(self, query, conversation_id=1, limit=4):
        terms = extract_terms(query)
        if not terms:
            return []
        
        with self.lock:
            # 1. Total indexed messages in this conversation
            total_docs_row = self.conn.execute(
                "SELECT COUNT(DISTINCT message_id) AS total FROM memory_index mi JOIN messages m ON m.id = mi.message_id WHERE m.conversation_id = ?",
                (conversation_id,)
            ).fetchone()
            total_docs = total_docs_row["total"] if total_docs_row else 0
            if total_docs == 0:
                return []
            
            # 2. Document frequency (DF) for each search term
            dfs = {}
            for term in terms:
                df_row = self.conn.execute(
                    """
                    SELECT COUNT(DISTINCT mi.message_id) AS df
                    FROM memory_index mi
                    JOIN messages m ON m.id = mi.message_id
                    WHERE mi.term = ? AND m.conversation_id = ?
                    """,
                    (term, conversation_id)
                ).fetchone()
                dfs[term] = df_row["df"] if df_row else 0

            # 3. Calculate IDF for each term
            import math
            idfs = {}
            for term, df in dfs.items():
                if df > 0:
                    idfs[term] = math.log(1.0 + (total_docs / df))
                else:
                    idfs[term] = 0.0

            # 4. Fetch the weights of each matched term in messages
            placeholders = ",".join("?" for _ in terms)
            rows = self.conn.execute(
                f"""
                SELECT m.id, m.role, m.content, mi.term, mi.weight
                FROM memory_index mi
                JOIN messages m ON m.id = mi.message_id
                WHERE mi.term IN ({placeholders}) AND m.conversation_id = ?
                """,
                (*terms, conversation_id)
            ).fetchall()

            # 5. Sum the TF-IDF score for each message
            message_scores = {}
            message_data = {}
            for row in rows:
                msg_id = row["id"]
                term = row["term"]
                weight = row["weight"]
                
                term_score = weight * idfs.get(term, 0.0)
                
                if msg_id not in message_scores:
                    message_scores[msg_id] = 0.0
                    message_data[msg_id] = {"role": row["role"], "content": row["content"]}
                message_scores[msg_id] += term_score

            # 6. Sort messages by TF-IDF score and return the top 'limit' ones
            sorted_msg_ids = sorted(message_scores.keys(), key=lambda k: message_scores[k], reverse=True)
            results = []
            for msg_id in sorted_msg_ids[:limit]:
                results.append(message_data[msg_id])
            
            return results

    def migrate_json_history(self, history_file, system_prompt):
        marker_key = "json_history_migrated"
        with self.lock:
            row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (marker_key,)).fetchone()
            count = self.conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
        if row or count > 0 or not history_file.exists():
            return

        try:
            data = json.loads(history_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.log_event("warning", "storage", "Ancien historique JSON illisible, migration ignorée.")
            return

        if not isinstance(data, list):
            return

        for item in data:
            role = item.get("role")
            content = item.get("content")
            if role == "system" or not role or not content:
                continue
            self.add_message(role, content)

        self.put_setting(marker_key, {"source": str(history_file), "system_prompt": system_prompt})
        self.log_event("info", "storage", f"Historique JSON migré depuis {history_file}.")

    # --- Plan Management ---

    def create_plan(self, conversation_id, query, steps):
        with self.lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO plans(conversation_id, query, status, created_at) VALUES (?, ?, 'active', ?)",
                (conversation_id, query, now_ts())
            )
            plan_id = cur.lastrowid
            
            step_tuples = [
                (plan_id, idx + 1, desc, 'pending', now_ts())
                for idx, desc in enumerate(steps)
            ]
            self.conn.executemany(
                "INSERT INTO plan_steps(plan_id, step_number, description, status, updated_at) VALUES (?, ?, ?, ?, ?)",
                step_tuples
            )
            return plan_id

    def get_active_plan(self, conversation_id):
        with self.lock:
            plan_row = self.conn.execute(
                "SELECT id, query, status FROM plans WHERE conversation_id = ? AND status = 'active' ORDER BY id DESC LIMIT 1",
                (conversation_id,)
            ).fetchone()
            
            if not plan_row:
                return None
                
            plan = dict(plan_row)
            steps_rows = self.conn.execute(
                "SELECT id, step_number, description, status, result_summary FROM plan_steps WHERE plan_id = ? ORDER BY step_number ASC",
                (plan["id"],)
            ).fetchall()
            
            plan["steps"] = [dict(r) for r in steps_rows]
            return plan

    def update_plan_step(self, step_id, status, result_summary=None):
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE plan_steps SET status = ?, result_summary = ?, updated_at = ? WHERE id = ?",
                (status, result_summary, now_ts(), step_id)
            )

    def mark_plan_completed(self, plan_id):
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE plans SET status = 'completed' WHERE id = ?",
                (plan_id,)
            )

    def cancel_active_plans(self, conversation_id):
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE plans SET status = 'cancelled' WHERE conversation_id = ? AND status = 'active'",
                (conversation_id,)
            )

    # --- ML Models Management ---

    def create_ml_model(self, name, description, algorithm, hyperparams):
        with self.lock, self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO ml_models(name, description, algorithm, hyperparams, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'draft', ?, ?)
                """,
                (name, description, algorithm, json.dumps(hyperparams, ensure_ascii=False), now_ts(), now_ts())
            )
            return cur.lastrowid

    def get_ml_models(self):
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, name, description, algorithm, hyperparams, status, created_at, updated_at FROM ml_models ORDER BY id DESC"
            ).fetchall()
        models = []
        for row in rows:
            model = dict(row)
            try:
                model["hyperparams"] = json.loads(model["hyperparams"]) if model["hyperparams"] else {}
            except json.JSONDecodeError:
                model["hyperparams"] = {}
            models.append(model)
        return models

    def get_ml_model(self, model_id):
        with self.lock:
            row = self.conn.execute(
                "SELECT id, name, description, algorithm, hyperparams, status, created_at, updated_at FROM ml_models WHERE id = ?",
                (model_id,)
            ).fetchone()
        if not row:
            return None
        model = dict(row)
        try:
            model["hyperparams"] = json.loads(model["hyperparams"]) if model["hyperparams"] else {}
        except json.JSONDecodeError:
            model["hyperparams"] = {}
        return model

    def get_ml_model_by_name(self, name):
        with self.lock:
            row = self.conn.execute(
                "SELECT id, name, description, algorithm, hyperparams, status, created_at, updated_at FROM ml_models WHERE name = ?",
                (name,)
            ).fetchone()
        if not row:
            return None
        model = dict(row)
        try:
            model["hyperparams"] = json.loads(model["hyperparams"]) if model["hyperparams"] else {}
        except json.JSONDecodeError:
            model["hyperparams"] = {}
        return model

    def update_ml_model(self, model_id, name, description, hyperparams, status=None):
        with self.lock, self.conn:
            if status:
                self.conn.execute(
                    """
                    UPDATE ml_models
                    SET name = ?, description = ?, hyperparams = ?, status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (name, description, json.dumps(hyperparams, ensure_ascii=False), status, now_ts(), model_id)
                )
            else:
                self.conn.execute(
                    """
                    UPDATE ml_models
                    SET name = ?, description = ?, hyperparams = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (name, description, json.dumps(hyperparams, ensure_ascii=False), now_ts(), model_id)
                )

    def delete_ml_model(self, model_id):
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM ml_models WHERE id = ?", (model_id,))

    def create_ml_model_version(self, model_id, version_tag, file_path, dataset_path, dataset_size, metrics, training_duration_ms, notes=None):
        with self.lock, self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO ml_model_versions(model_id, version_tag, file_path, dataset_path, dataset_size, metrics, training_duration_ms, is_active, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (model_id, version_tag, file_path, dataset_path, dataset_size, json.dumps(metrics, ensure_ascii=False), training_duration_ms, notes, now_ts())
            )
            return cur.lastrowid

    def get_ml_model_versions(self, model_id):
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, model_id, version_tag, file_path, dataset_path, dataset_size, metrics, training_duration_ms, is_active, notes, created_at FROM ml_model_versions WHERE model_id = ? ORDER BY id DESC",
                (model_id,)
            ).fetchall()
        versions = []
        for row in rows:
            version = dict(row)
            try:
                version["metrics"] = json.loads(version["metrics"]) if version["metrics"] else {}
            except json.JSONDecodeError:
                version["metrics"] = {}
            versions.append(version)
        return versions

    def get_ml_model_version(self, version_id):
        with self.lock:
            row = self.conn.execute(
                "SELECT id, model_id, version_tag, file_path, dataset_path, dataset_size, metrics, training_duration_ms, is_active, notes, created_at FROM ml_model_versions WHERE id = ?",
                (version_id,)
            ).fetchone()
        if not row:
            return None
        version = dict(row)
        try:
            version["metrics"] = json.loads(version["metrics"]) if version["metrics"] else {}
        except json.JSONDecodeError:
            version["metrics"] = {}
        return version

    def get_active_ml_model_version(self, model_id):
        with self.lock:
            row = self.conn.execute(
                "SELECT id, model_id, version_tag, file_path, dataset_path, dataset_size, metrics, training_duration_ms, is_active, notes, created_at FROM ml_model_versions WHERE model_id = ? AND is_active = 1 LIMIT 1",
                (model_id,)
            ).fetchone()
        if not row:
            return None
        version = dict(row)
        try:
            version["metrics"] = json.loads(version["metrics"]) if version["metrics"] else {}
        except json.JSONDecodeError:
            version["metrics"] = {}
        return version

    def activate_ml_model_version(self, model_id, version_id):
        with self.lock, self.conn:
            self.conn.execute("UPDATE ml_model_versions SET is_active = 0 WHERE model_id = ?", (model_id,))
            self.conn.execute("UPDATE ml_model_versions SET is_active = 1 WHERE id = ?", (version_id,))
            self.conn.execute("UPDATE ml_models SET status = 'active', updated_at = ? WHERE id = ?", (now_ts(), model_id))

    def delete_ml_model_version(self, version_id):
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM ml_model_versions WHERE id = ?", (version_id,))

    def add_ml_training_history(self, model_id, version_id, action, details):
        with self.lock, self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO ml_training_history(model_id, version_id, action, details, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (model_id, version_id, action, json.dumps(details, ensure_ascii=False), now_ts())
            )
            return cur.lastrowid

    def get_ml_training_history(self, model_id):
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, model_id, version_id, action, details, created_at FROM ml_training_history WHERE model_id = ? ORDER BY id DESC",
                (model_id,)
            ).fetchall()
        history = []
        for row in rows:
            item = dict(row)
            try:
                item["details"] = json.loads(item["details"]) if item["details"] else {}
            except json.JSONDecodeError:
                item["details"] = {}
            history.append(item)
        return history

    # --- Agent Management ---

    def register_agent(self, name, role, status="stopped", interval=None):
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO agents(name, role, status, interval, last_seen)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    role = excluded.role,
                    interval = excluded.interval,
                    last_seen = excluded.last_seen
                """,
                (name, role, status, interval, now_ts()),
            )

    def update_agent_status(self, name, status):
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE agents SET status = ?, last_seen = ? WHERE name = ?",
                (status, now_ts(), name),
            )

    def get_agents(self):
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, name, role, status, interval, last_seen FROM agents ORDER BY name"
            ).fetchall()
        return [dict(row) for row in rows]

    # --- Chat Sessions (API SSE) ---
    # Persistance légère du statut des sessions de chat en cours, pour survivre
    # à un redémarrage du serveur (audit / statut), même si le flux SSE en
    # lui-même ne peut pas être repris après coup.

    def create_chat_session(self, task_id, conversation_id, agent_mode, model_id, status="pending"):
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO chat_sessions(task_id, conversation_id, agent_mode, model_id, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET status = excluded.status, updated_at = excluded.updated_at
                """,
                (task_id, conversation_id, agent_mode, model_id, status, now_ts(), now_ts()),
            )

    def update_chat_session_status(self, task_id, status):
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE chat_sessions SET status = ?, updated_at = ? WHERE task_id = ?",
                (status, now_ts(), task_id),
            )

    def get_chat_session(self, task_id):
        with self.lock:
            row = self.conn.execute(
                "SELECT task_id, conversation_id, agent_mode, model_id, status, created_at, updated_at "
                "FROM chat_sessions WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return dict(row) if row else None

    def mark_stale_chat_sessions_interrupted(self):
        """
        À appeler au démarrage du serveur : toute session encore marquée
        active en base provient nécessairement d'un arrêt précédent (crash
        ou redémarrage), puisque les sessions vivent en mémoire.
        """
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE chat_sessions SET status = 'interrupted', updated_at = ? "
                "WHERE status IN ('pending', 'running', 'waiting_approval')",
                (now_ts(),),
            )

