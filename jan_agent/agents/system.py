import threading
from pathlib import Path
from .base import BackgroundAgent
from ..utils import now_ts
from ..config import load_config

class PerformanceAgent(BackgroundAgent):
    def __init__(self, store, stop_event, logger):
        super().__init__("performance-agent", store, stop_event, logger, interval=10)

    def tick(self):
        active_threads = threading.active_count()
        active_messages = self.store.count_active_messages()
        self.store.put_metric("active_threads", active_threads)
        self.store.put_metric("active_messages", active_messages)
        try:
            db_size_kb = self.store.path.stat().st_size / 1024
            self.store.put_metric("db_size_kb", db_size_kb)
        except OSError:
            db_size_kb = 0
        self.logger.log(
            self.name,
            f"threads={active_threads}, messages={active_messages}, db={db_size_kb:.1f} Ko",
            "all",
        )


class ConfigAgent(BackgroundAgent):
    def __init__(self, store, stop_event, logger, config_path, on_reload):
        super().__init__("config-agent", store, stop_event, logger, interval=2)
        self.config_path = Path(config_path)
        self.on_reload = on_reload
        self.last_mtime = self._mtime()

    def _mtime(self):
        try:
            return self.config_path.stat().st_mtime
        except OSError:
            return None

    def tick(self):
        current = self._mtime()
        if current is None or current == self.last_mtime:
            return
        self.last_mtime = current
        cfg = load_config(self.config_path)
        self.on_reload(cfg)
        self.store.put_setting("last_config_reload", {"path": str(self.config_path), "at": now_ts()})
        self.store.log_event("info", self.name, "Configuration rechargée dynamiquement.")
        self.logger.log(self.name, "configuration rechargée", "debug")
