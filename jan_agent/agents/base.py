import threading

class BackgroundAgent(threading.Thread):
    def __init__(self, name, store, stop_event, logger, interval=2):
        super().__init__(name=name, daemon=True)
        self.store = store
        self.stop_event = stop_event
        self.logger = logger
        self.interval = interval

    def run(self):
        self.store.log_event("info", self.name, "Agent démarré.")
        self.logger.log(self.name, "démarré", "debug")
        while not self.stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:
                self.store.log_event("error", self.name, str(exc))
                self.logger.log(self.name, f"erreur : {exc}", "debug")
            self.stop_event.wait(self.interval)
        self.store.log_event("info", self.name, "Agent arrêté.")
        self.logger.log(self.name, "arrêté", "all")

    def tick(self):
        raise NotImplementedError
