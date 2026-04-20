#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


HEARTBEAT_FILE = Path(os.getenv("RUNNER_HEARTBEAT_FILE", "data/runner_heartbeat.json"))
MAX_AGE_SECONDS = int(os.getenv("RUNNER_HEARTBEAT_MAX_AGE_SECONDS", "180"))
UNHEALTHY_STATES = {"safe_mode", "error"}


def main() -> int:
    if not HEARTBEAT_FILE.exists():
        print(f"missing heartbeat file: {HEARTBEAT_FILE}")
        return 1

    try:
        payload = json.loads(HEARTBEAT_FILE.read_text())
    except Exception as exc:
        print(f"invalid heartbeat file: {exc}")
        return 1

    updated_at = payload.get("updated_at")
    status = payload.get("status", "unknown")
    if not updated_at:
        print("heartbeat missing updated_at")
        return 1

    try:
        last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception as exc:
        print(f"invalid heartbeat timestamp: {exc}")
        return 1

    age = (datetime.now(timezone.utc) - last_update).total_seconds()
    if age > MAX_AGE_SECONDS:
        print(f"stale heartbeat: age={age:.1f}s max={MAX_AGE_SECONDS}s")
        return 1

    if status in UNHEALTHY_STATES:
        print(f"runner unhealthy: status={status}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
