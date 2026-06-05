import threading
from typing import Optional


class WorkerPool:
    """Thread-safe round-robin URL pool with health tracking."""

    def __init__(self, urls: list[str]):
        self._urls = list(urls)
        self._healthy = {u: True for u in urls}
        self._idx = 0
        self._lock = threading.Lock()

    def next(self) -> Optional[str]:
        with self._lock:
            healthy = [u for u in self._urls if self._healthy[u]]
            if not healthy:
                return None
            url = healthy[self._idx % len(healthy)]
            self._idx += 1
            return url

    def mark_unhealthy(self, url: str) -> None:
        self._healthy[url] = False

    def mark_healthy(self, url: str) -> None:
        self._healthy[url] = True

    def status(self) -> list[dict]:
        return [{"url": u, "healthy": self._healthy[u]} for u in self._urls]
