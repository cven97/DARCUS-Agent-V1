import json
from .base import BackgroundAgent

class TaskExecutionAgent(BackgroundAgent):
    def __init__(self, store, stop_event, logger, executor, result_bus):
        super().__init__("task-agent", store, stop_event, logger, interval=0.5)
        self.executor = executor
        self.result_bus = result_bus

    def tick(self):
        task = self.store.fetch_pending_task()
        if not task:
            return
        self.logger.log(self.name, f"prise en charge tâche #{task['id']} ({task['kind']})", "debug")

        try:
            payload = json.loads(task["payload"])
        except json.JSONDecodeError as exc:
            self._finish(task["id"], None, f"Payload invalide : {exc}")
            return

        if task["kind"] != "command":
            self._finish(task["id"], None, f"Type de tâche inconnu : {task['kind']}")
            return

        code, stdout, stderr = self.executor(payload.get("command", ""))
        self.logger.log(self.name, f"commande terminée avec code {code}", "debug")
        result = json.dumps(
            {"code": code, "stdout": stdout, "stderr": stderr},
            ensure_ascii=False,
        )
        self._finish(task["id"], result, None if code == 0 else stderr)

    def _finish(self, task_id, result, error):
        self.store.finish_task(task_id, result=result, error=error)
        waiter = self.result_bus.pop(task_id, None)
        if waiter:
            waiter.put(self.store.get_task_result(task_id))
