import queue
import threading
from .agents import (
    MemoryAgent,
    RetrievalAgent,
    PerformanceAgent,
    ConfigAgent,
    TaskExecutionAgent,
)

class AgentRuntime:
    def __init__(self, store, config_path, on_reload, command_executor, logger):
        self.store = store
        self.stop_event = threading.Event()
        self.result_bus = {}
        self.result_lock = threading.Lock()
        self.logger = logger
        self.agents = [
            MemoryAgent(store, self.stop_event, logger),
            RetrievalAgent(store, self.stop_event, logger),
            PerformanceAgent(store, self.stop_event, logger),
            ConfigAgent(store, self.stop_event, logger, config_path, on_reload),
            TaskExecutionAgent(store, self.stop_event, logger, command_executor, self.result_bus),
        ]

    def start(self):
        for agent in self.agents:
            agent.start()

    def stop(self):
        self.stop_event.set()
        for agent in self.agents:
            agent.join(timeout=2)

    def submit_command(self, command, timeout=120):
        task_id = self.store.create_task("command", {"command": command}, status="queued")
        self.logger.log("runtime", f"tâche #{task_id} créée pour task-agent", "debug")
        waiter = queue.Queue(maxsize=1)
        with self.result_lock:
            self.result_bus[task_id] = waiter
        self.store.set_task_status(task_id, "pending")

        current = self.store.get_task_result(task_id)
        if current and current["status"] in {"done", "failed"}:
            with self.result_lock:
                self.result_bus.pop(task_id, None)
            return current

        try:
            result = waiter.get(timeout=timeout)
        except queue.Empty:
            with self.result_lock:
                self.result_bus.pop(task_id, None)
            return {"status": "failed", "result": None, "error": "Tâche expirée."}
        return result
