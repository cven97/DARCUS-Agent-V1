# Documentation de l'API

L'API DARCUS Agent est une application FastAPI. Lancez-la avec :

```bash
export JWT_SECRET="votre-secret"
python3 main.py --api          # http://localhost:8000
```

Une documentation interactive (Swagger UI) est générée automatiquement sur **`/docs`**, et le schéma OpenAPI brut sur **`/openapi.json`**. Ce document en donne la vue d'ensemble et les usages courants.

## Sommaire

- [Authentification](#authentification)
- [Conventions](#conventions)
- [Conversations et chat](#conversations-et-chat)
- [Chat en streaming (SSE)](#chat-en-streaming-sse)
- [Configuration et supervision](#configuration-et-supervision)
- [Modèles ML](#modèles-ml)
- [Générateur de données](#générateur-de-données)
- [Codes d'erreur](#codes-derreur)
- [Points d'attention](#points-dattention)

## Authentification

Les routes marquées 🔒 exigent un jeton JWT dans l'en-tête :

```
Authorization: Bearer <jeton>
```

- Algorithme : **HS256**, signé avec la variable d'environnement `JWT_SECRET`.
- L'API **ne délivre pas** de jetons : ils sont émis par le service d'authentification, qui doit utiliser le même secret.
- Le champ `sub` du jeton identifie l'utilisateur. Il sert de propriétaire aux conversations.
- Jeton absent, invalide ou expiré : `401` (`Token invalide` ou `Token a expiré`).

Pour tester sans service d'authentification, générez un jeton avec le même secret :

```bash
python3 - <<'EOF'
import os, time, jwt
print(jwt.encode({"sub": "test", "exp": int(time.time()) + 3600},
                 os.environ["JWT_SECRET"], algorithm="HS256"))
EOF
```

## Conventions

- Corps de requête et de réponse en **JSON** (sauf indication contraire).
- Erreurs au format FastAPI : `{"detail": "message"}`.
- Les exemples utilisent `$TOKEN` pour le jeton et `http://localhost:8000` comme base.

---

## Conversations et chat

### État du serveur

`GET /` → `{"status": "online", "message": "JanAgent API is running."}`

### Conversations 🔒

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/api/conversations` | Liste les conversations de l'utilisateur du jeton |
| `POST` | `/api/conversations` | Crée une conversation |
| `DELETE` | `/api/conversations/{conv_id}` | Supprime une conversation de l'utilisateur |

`POST /api/conversations` — corps :

| Champ | Type | Défaut |
|---|---|---|
| `title` | string | `"Nouvelle Conversation"` |

```bash
curl -X POST localhost:8000/api/conversations \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"title": "Mon premier échange"}'
# {"id": 1, "title": "Mon premier échange"}
```

### Historique, remise à zéro, plan 🔒

| Méthode | Route | Réponse |
|---|---|---|
| `GET` | `/api/chat/{conversation_id}/history` | Liste de messages `{id, role, content, created_at, archived}`, du plus ancien au plus récent |
| `POST` | `/api/chat/{conversation_id}/reset` | `{"status": "success", "message": "Conversation history reset."}` |
| `GET` | `/api/chat/{conversation_id}/plan` | Le plan d'action actif, ou `{"status": "no_active_plan"}` |

---

## Chat en streaming (SSE)

Un échange se déroule en **trois temps** : démarrer une session, écouter le flux, répondre aux demandes d'approbation.

### 1. Démarrer — `POST /api/v2/chat/send` 🔒

Corps :

| Champ | Type | Requis | Description |
|---|---|---|---|
| `message` | string | oui | Message de l'utilisateur (non vide) |
| `conversation_id` | integer | non | Conversation existante. Si absent, une nouvelle est créée, titrée d'après le message |
| `agent_mode` | string | non | `"loop"` (défaut, boucle ReAct avec outils) ou `"generative"` (génération directe) |
| `model_id` | string | non | Modèle LLM à utiliser. Voir `GET /api/v2/chat/models` |

Réponse :

```json
{
  "task_id": "…",
  "conversation_id": 1,
  "stream_url": "/api/v2/chat/stream/…",
  "status": "started"
}
```

L'agent s'exécute en arrière-plan ; la réponse revient immédiatement.

### 2. Écouter — `GET /api/v2/chat/stream/{task_id}`

Flux `text/event-stream`. Chaque événement est une ligne `data: <json>` suivie d'une ligne vide. Un commentaire `: heartbeat` est envoyé toutes les 30 s d'inactivité pour garder la connexion ouverte.

| `type` | Contenu | Signification |
|---|---|---|
| `connected` | `task_id` | Connexion établie (premier événement) |
| `event` | `event`, `content` | Événement de l'agent (voir ci-dessous) |
| `action_required` | `tool`, `args` | L'agent demande l'approbation d'une commande |
| `done` | – | Fin normale du traitement |
| `error` | `content` | Erreur, ou session annulée |

Valeurs de `event` pour les messages de `type: "event"` :

| `event` | Description |
|---|---|
| `thought`, `thought_delta` | Réflexion de l'agent (complète, ou par fragments) |
| `tool_requested`, `tool_executed` | Outil demandé, puis exécuté |
| `observation_received` | Résultat renvoyé au LLM |
| `response`, `response_delta` | Réponse finale (complète, ou par fragments) |
| `security_error` | Commande refusée par les règles de sécurité |
| `info` | Information de progression (classification, critique…) |

Le flux se ferme après `done` ou `error`.

```bash
curl -N localhost:8000/api/v2/chat/stream/$TASK_ID
```

```js
const es = new EventSource(`/api/v2/chat/stream/${taskId}`);
es.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "action_required") askUser(msg.tool, msg.args);
  if (msg.type === "done" || msg.type === "error") es.close();
};
```

### 3. Approuver — `POST /api/v2/chat/approve/{task_id}`

Répond à un événement `action_required`. Corps :

```json
{ "choice": "y" }
```

| `choice` | Effet |
|---|---|
| `y` | Autoriser cette exécution |
| `n` | Refuser |
| `a` | Autoriser et toujours approuver cette commande (ajoutée aux commandes auto-approuvées) |

Réponse : `{"status": "approved", "choice": "y"}`.
Erreurs : `400` (choix invalide), `404` (session inconnue), `409` (aucune approbation en attente, ou réponse déjà envoyée).

### Statut et annulation

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/api/v2/chat/status/{task_id}` | `{"task_id", "status"}`. Statuts : `pending`, `running`, `waiting_approval`, `done`, `error`, `cancelled`, ou `not_found`. Une session terminée ou perdue après redémarrage est relue en base |
| `POST` | `/api/v2/chat/cancel/{task_id}` | Annule la session : `{"status": "cancelled", "task_id": "…"}` |

### Modèles disponibles — `GET /api/v2/chat/models` 🔒

Retourne la liste des modèles du serveur LLM configuré. `502` si le serveur est injoignable.

### Exemple complet

```bash
# 1. démarrer
RESP=$(curl -s -X POST localhost:8000/api/v2/chat/send \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"message": "Liste les fichiers du dossier courant"}')
TASK_ID=$(echo "$RESP" | python3 -c 'import sys,json; print(json.load(sys.stdin)["task_id"])')

# 2. écouter (dans un terminal)
curl -N localhost:8000/api/v2/chat/stream/$TASK_ID

# 3. si un événement action_required apparaît, approuver
curl -X POST localhost:8000/api/v2/chat/approve/$TASK_ID \
  -H "Content-Type: application/json" -d '{"choice": "y"}'
```

---

## Configuration et supervision

Toutes ces routes exigent 🔒.

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/api/config` | Configuration courante du modèle |
| `POST` | `/api/config` | Met à jour la configuration |
| `GET` | `/api/agent/status` | Supervision : agents (`name`, `is_alive`, `interval`, `type`), métriques et logs système |
| `GET` | `/api/tasks` | Tâches d'arrière-plan enregistrées |
| `GET` | `/api/security/config` | Règles de sécurité des commandes |
| `POST` | `/api/security/auto-approve` | Ajoute une commande à l'auto-approbation |

`POST /api/config` — corps :

| Champ | Type | Requis |
|---|---|---|
| `server_url` | string | oui |
| `default_model` | string | oui |
| `temperature` | number | oui |
| `max_tokens` | integer | oui |
| `system_prompt` | string | non |
| `memory_prompt` | string | non |
| `planner_prompt` | string | non |

`POST /api/security/auto-approve` — corps : `{"command_base": "git"}`.

---

## Modèles ML

Cycle de vie de classifieurs de texte scikit-learn : création, entraînement, versions, évaluation, prédiction. Préfixe : `/api/models`.

Algorithmes supportés : `LogisticRegression`, `RandomForestClassifier`, `LinearSVC`, `MultinomialNB` (liste live : `GET /api/models/algorithms`).

| Méthode | Route | Auth | Description |
|---|---|---|---|
| `GET` | `/api/models/algorithms` | 🔒 | Algorithmes disponibles |
| `GET` | `/api/models` | 🔒 | Liste des modèles |
| `POST` | `/api/models` | 🔒 | Crée un modèle (`201`) |
| `GET` | `/api/models/{model_id}` | 🔒 | Détail d'un modèle |
| `PUT` | `/api/models/{model_id}` | 🔒 | Modifie nom, description, hyperparamètres |
| `DELETE` | `/api/models/{model_id}` | 🔒 | Supprime un modèle |
| `POST` | `/api/models/{model_id}/train` | 🔒 | Lance un entraînement en tâche de fond (`202`) |
| `GET` | `/api/models/{model_id}/versions` | 🔒 | Versions du modèle |
| `POST` | `/api/models/{model_id}/versions/{version_id}/activate` | 🔒 | Active une version |
| `DELETE` | `/api/models/{model_id}/versions/{version_id}` | 🔒 | Supprime une version |
| `POST` | `/api/models/{model_id}/evaluate` | 🔒 | Évalue le modèle sur un jeu de données |
| `GET` | `/api/models/{model_id}/history` | 🔒 | Historique des entraînements |
| `GET` | `/api/models/{model_id}/benchmark` | 🔒 | Mesure de performance (`samples`, optionnel) |
| `GET` | `/api/models/{model_id}/compare` | 🔒 | Compare les versions (`dataset_path`, requis) |
| `POST` | `/api/models/{model_id}/predict` | – | Prédit pour un texte |
| `POST` | `/api/models/{model_id}/predict/batch` | – | Prédit pour plusieurs textes |

Corps des requêtes :

| Route | Champs |
|---|---|
| `POST /api/models` | `name` (requis), `algorithm`, `description`, `hyperparams` (objet) |
| `PUT /api/models/{id}` | `name` (requis), `description`, `hyperparams` |
| `POST …/train` | `dataset_path` et `version_tag` (requis), `notes` |
| `POST …/evaluate` | `dataset_path` (requis) |
| `POST …/predict` | `text` (requis) |
| `POST …/predict/batch` | `texts` : tableau de chaînes (requis) |

```bash
curl -X POST localhost:8000/api/models \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"name": "sentiment", "algorithm": "LogisticRegression"}'

curl -X POST localhost:8000/api/models/1/predict \
  -H "Content-Type: application/json" -d '{"text": "Excellent produit"}'
```

---

## Générateur de données

Génère des jeux de données de texte étiquetés à l'aide du LLM, les stocke dans PostgreSQL et les exporte. Nécessite la configuration de `pr_generator/data_config.json`.

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/api/generate` | Lance une génération en tâche de fond (`202`), retourne `{"task_id"}` |
| `GET` | `/api/task/{tid}` | Statut d'une tâche (dates de création et de fin incluses) |
| `GET` | `/api/projects` | Liste des projets |
| `GET` | `/api/data/{project_name}` | Données d'un projet |
| `PUT` | `/api/data/{data_id}` | Modifie une entrée (`text`, `label`) |
| `DELETE` | `/api/data/delete/{data_id}` | Supprime une entrée |
| `POST` | `/api/data/augment/{data_id}` | Génère des variantes d'une entrée |
| `GET` | `/api/export/{project_name}?format=json` | Exporte un projet (`json` par défaut, ou `csv`) |
| `POST` | `/api/import/{project_name}` | Importe un fichier (formulaire multipart, champ `file`) — `201` |
| `GET` | `/api/templates` | Liste les templates de prompt |
| `GET` | `/api/templates/{name}` | Lit un template |
| `POST` | `/api/templates/{name}` | Enregistre un template (`{"template": "…"}`) — `201` |

`POST /api/generate` — corps :

| Champ | Type | Requis | Description |
|---|---|---|---|
| `project` | string | oui | Nom du projet |
| `topic` | string | oui | Sujet des textes à générer |
| `label` | string | oui | Étiquette attribuée |
| `count` | integer | non | Nombre de textes |
| `template` | string | non | Template de prompt (défaut : `default`) |
| `variables` | string[] | non | Variables injectées dans le template |

---

## Codes d'erreur

| Code | Cas |
|---|---|
| `400` | Requête invalide (message vide, choix d'approbation inconnu…) |
| `401` | Jeton JWT absent, invalide ou expiré |
| `404` | Session ou ressource introuvable |
| `409` | Conflit d'état (aucune approbation en attente) |
| `422` | Corps ou paramètres ne respectant pas le schéma |
| `500` | Erreur interne (le détail est dans `detail`) |
| `502` | Serveur LLM injoignable |

## Points d'attention

Ces comportements sont ceux du code actuel ; à connaître avant d'exposer l'API.

- **Routes sans JWT.** Le générateur de données (`/api/generate`, `/api/data/…`, `/api/export/…`, `/api/import/…`, `/api/templates/…`, `/api/task/…`, `/api/projects`), les routes `predict` de `/api/models`, et `stream`, `approve`, `status` et `cancel` de `/api/v2/chat` sont **publiques**. Pour le chat, le `task_id` joue le rôle de secret : ne le partagez pas. Ne publiez pas l'API sur un réseau non fiable sans un proxy qui authentifie ces routes.
- **Propriété des conversations.** `history`, `reset` et `plan` filtrent par identifiant de conversation, sans vérifier que celle-ci appartient à l'utilisateur du jeton.
- **Approbation de commandes.** Une commande hors liste d'auto-approbation attend une réponse via `approve`. Sans réponse au bout de **5 minutes**, elle est refusée (équivalent de `n`) et la session reprend.
