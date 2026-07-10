from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def request_json(url: str, payload: dict | None = None) -> dict:
    if payload is None:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    args = parser.parse_args()
    health = request_json(args.base_url.rstrip("/") + "/health")
    print("health:", json.dumps(health, ensure_ascii=False))
    if health.get("status") != "ok":
        raise SystemExit(1)
    payload = {
        "message": "I have crushing chest pain, severe shortness of breath, and cold sweating",
        "history": [],
        "language": "en",
        "source": "local_smoke",
    }
    chat = request_json(args.base_url.rstrip("/") + "/chat", payload)
    print("chat:", json.dumps({k: chat.get(k) for k in ["mode", "urgency_level", "suggested_doctor", "possible_diagnosis"]}, ensure_ascii=False))
    if chat.get("mode") != "emergency" or chat.get("urgency_level") != "High" or "Emergency" not in str(chat.get("suggested_doctor")):
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
