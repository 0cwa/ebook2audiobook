"""Small protocol fixture used by dependency-light Chatterbox client tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


SUPPORTED_LANGUAGES = [
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh",
]


def send(message):
    print(json.dumps(message, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--approved-root", action="append", default=[])
    parser.add_argument("--approved-voice-root", action="append", default=[])
    parser.add_argument("--approved-output-root", action="append", default=[])
    parser.add_argument("--mode", default="success")
    args = parser.parse_args()
    mode = args.mode
    if mode == "no-ready":
        time.sleep(30)
        return 0
    send({"protocol": 1, "event": "ready", "device": "cpu", "languages": SUPPORTED_LANGUAGES, "sample_rate": 24000})
    for line in os.sys.stdin:
        request = json.loads(line)
        operation = request.get("op")
        if operation == "ping":
            send({"protocol": 1, "id": request.get("id"), "ok": True, "result": {"device": "cpu", "sample_rate": 24000}})
        elif operation == "shutdown":
            send({"protocol": 1, "id": request.get("id"), "ok": True, "result": {}})
            return 0
        elif operation == "cancel":
            send({"protocol": 1, "id": request.get("id"), "ok": True, "result": {"cancel_requested": True}})
        elif operation == "synthesize":
            if mode == "malformed":
                print("{malformed", flush=True)
                return 0
            if mode == "crash":
                os._exit(17)
            if mode == "partial-slow":
                output = request["output"]["path"]
                if output.endswith(".part"):
                    output = output[:-5]
                path = Path(output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.parent.joinpath(f".{path.name}.{os.getpid()}.fixture.part").write_bytes(b"partial")
                time.sleep(30)
            if mode in {"slow", "cancel-wait"}:
                time.sleep(30)
            output = request["output"]["path"]
            if output.endswith(".part"):
                output = output[:-5]
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake-flac")
            send({
                "protocol": 1,
                "id": request.get("id"),
                "ok": True,
                "result": {
                    "path": str(path),
                    "sample_rate": 24000,
                    "channels": 1,
                    "language": request.get("language"),
                    "sha256": __import__("hashlib").sha256(b"fake-flac").hexdigest(),
                },
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
