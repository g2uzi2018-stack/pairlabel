"""Serialize edits, exports and recycling within this single-worker deployment."""
import threading
from contextlib import contextmanager

_locks = {}


@contextmanager
def job_guard(jid):
    lock = _locks.setdefault(jid, threading.RLock())
    with lock:
        yield
