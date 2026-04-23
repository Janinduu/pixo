"""Airgap enforcement — block outbound network calls during a run.

When active, any attempt to open an AF_INET or AF_INET6 socket to a non-loopback
address raises AirgapViolation. Loopback traffic (127.0.0.1, ::1) is permitted
so local services keep working.

Two-layer defense:
  1. Hint layer — set env vars / patch known library connectivity checks so
     libraries that respect "offline mode" skip the call entirely.
  2. Enforcement layer — monkey-patch sockets so anything that still tries
     to dial out fails fast with AirgapViolation.

Usage:
    with airgap_enforced():
        ...  # any socket to the internet raises AirgapViolation
"""

import os
import socket
from contextlib import contextmanager


class AirgapViolation(RuntimeError):
    """Raised when code tries to connect to the network in airgap mode."""


_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}

# Env vars that widely-used ML libraries respect to stay offline.
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",          # huggingface-hub
    "TRANSFORMERS_OFFLINE": "1",     # transformers
    "HF_DATASETS_OFFLINE": "1",      # datasets
    "YOLO_OFFLINE": "True",          # ultralytics (historical)
    "YOLO_VERBOSE": "False",         # ultralytics (quiet)
}


def _is_loopback(address) -> bool:
    if address is None:
        return True
    if isinstance(address, tuple):
        host = address[0] if address else ""
    else:
        host = str(address)
    if not host:
        return True
    return host in _LOOPBACK or host.startswith("127.") or host.startswith("::1")


def _silence_ultralytics_checks():
    """Patch ultralytics' online check so it doesn't ping 1.1.1.1.

    Returns a restore() callable, or None if ultralytics is not importable.
    """
    try:
        from ultralytics.utils import checks as ul_checks
    except Exception:
        return None

    restores = []

    # check_online() pings 1.1.1.1 — make it silently report "offline".
    if hasattr(ul_checks, "check_online"):
        original = ul_checks.check_online
        ul_checks.check_online = lambda: False  # type: ignore[assignment]
        restores.append(lambda: setattr(ul_checks, "check_online", original))

    # Disable the background settings sync that also reaches out.
    try:
        from ultralytics.utils import SETTINGS  # type: ignore
        prev_sync = SETTINGS.get("sync", True)
        SETTINGS["sync"] = False
        restores.append(lambda: SETTINGS.__setitem__("sync", prev_sync))
    except Exception:
        pass

    def restore():
        for r in restores:
            try:
                r()
            except Exception:
                pass

    return restore


@contextmanager
def airgap_enforced():
    """Block non-loopback network access for the duration of the context."""
    # --- Layer 1: hint libraries that we're offline ---
    prev_env = {k: os.environ.get(k) for k in _OFFLINE_ENV}
    os.environ.update(_OFFLINE_ENV)
    restore_ultralytics = _silence_ultralytics_checks()

    # --- Layer 2: hard-block any remaining outbound sockets/DNS ---
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address):
            raise AirgapViolation(
                f"Airgap mode: blocked outbound connection to {address}. "
                f"Run without --airgap to allow network access."
            )
        return original_connect(self, address)

    def guarded_connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address):
            raise AirgapViolation(
                f"Airgap mode: blocked outbound connection to {address}."
            )
        return original_connect_ex(self, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host and not _is_loopback(host):
            raise AirgapViolation(
                f"Airgap mode: blocked DNS lookup for '{host}'."
            )
        return original_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = guarded_getaddrinfo
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.getaddrinfo = original_getaddrinfo
        if restore_ultralytics:
            restore_ultralytics()
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
