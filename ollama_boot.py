"""Start the local Ollama server when a translation step needs it."""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time

HOST = "127.0.0.1"
PORT = 11434


def ollama_running() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=0.5):
            return True
    except OSError:
        return False


def _try_start(command: list[str], detach: bool) -> bool:
    if detach:
        if shutil.which(command[0]) is None:
            return False
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    result = subprocess.run(command, capture_output=True)
    return result.returncode == 0


def ensure_ollama(timeout: float = 25) -> bool:
    """Return True when Ollama answers on port 11434, starting it if needed."""
    if ollama_running():
        return True

    print("⚠️ Ollama가 꺼져 있습니다. 실행을 시도합니다...", flush=True)
    attempts: list[tuple[list[str], bool, float]] = []
    if sys.platform == "darwin":
        attempts.append((["open", "-a", "Ollama"], False, 12))
    attempts.append((["ollama", "serve"], True, timeout))

    deadline = time.monotonic() + timeout
    launched = False
    for command, detach, wait in attempts:
        if time.monotonic() >= deadline:
            break
        if not _try_start(command, detach):
            continue
        launched = True
        attempt_deadline = min(deadline, time.monotonic() + wait)
        while time.monotonic() < attempt_deadline:
            if ollama_running():
                print("✓ Ollama가 실행되었습니다.", flush=True)
                return True
            time.sleep(0.5)

    if launched:
        print("⚠️ Ollama가 시간 안에 켜지지 않았습니다.", flush=True)
    else:
        print("⚠️ Ollama 앱과 'ollama' 명령을 찾지 못했습니다.", flush=True)
    return False
