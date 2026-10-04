"""Package S1 acceptance script: seed a learner, reassess, reach STABLE.

Runs against the real app in-process (temp database) or against a server you started:

    .venv\\Scripts\\python -m tests.S1.demo_flow
    .venv\\Scripts\\python -m tests.S1.demo_flow --url http://127.0.0.1:8000

The same `run()` is called by tests/S1/test_s1.py. It uses the always-reachable
`POST /learner/{id}/intervene` (see notes/S1.md for why `POST /intervene` may be answered by
the older stateless route) and falls back gracefully if the server has no interpreter.
"""
import argparse
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

P03_LE = """int total_energy(int cells[], int n) {
    int total = 0;
    for (int i = 0; i <= n; i++) {
        total += cells[i];
    }
    return total;
}"""


class UrlClient:
    """post/get with the same (status, body) shape as the in-process client below."""

    def __init__(self, base):
        self.base = base.rstrip("/")

    def _call(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    def post(self, path, body=None):
        return self._call("POST", path, body or {})

    def get(self, path):
        return self._call("GET", path)


class AppClient:
    def __init__(self, test_client):
        self.client = test_client

    def post(self, path, body=None):
        response = self.client.post(path, json=body or {})
        return response.status_code, response.json()

    def get(self, path):
        response = self.client.get(path)
        return response.status_code, response.json()


def run(client, say=print):
    """The flow. Returns the final /reassess body. Raises AssertionError at the first wrong step."""
    def step(label, result, expect=200):
        status, body = result
        say(f"{label}: HTTP {status}")
        assert status == expect, f"{label}: expected {expect}, got {status}: {body}"
        return body

    created = step("POST /learner", client.post("/learner", {"callsign": "NOVA"}))
    learner_id = created["learner_id"]
    say(f"  learner_id = {learner_id}")

    seeded = step(f"POST /learner/{learner_id}/seed", client.post(f"/learner/{learner_id}/seed"))
    states = {cls: seeded["misconceptions"][cls]["state"] for cls in ("M01", "M06", "M08")}
    say(f"  M01/M06/M08 = {states}, {len(seeded['attempts'])} attempts")
    assert states == {"M01": "STABLE", "M06": "MASTERED", "M08": "ACTIVE"}

    fetched = step(f"GET /learner/{learner_id}", client.get(f"/learner/{learner_id}"))
    assert fetched["misconceptions"]["M08"]["p_active"] == 0.9

    # The intervention is optional for the reassess below (a reassess item on an ACTIVE class counts it as
    # taken), but with it the TREATING -> PROBATION steps are real.
    status, package = client.post(f"/learner/{learner_id}/intervene",
                                  {"class": "M08", "problem_id": "P03", "code": P03_LE})
    if status == 200:
        say(f"POST /learner/{learner_id}/intervene: modality={package['modality']} "
            f"state={package['learner_state']['state']}")
        assert package["learner_state"]["state"] == "TREATING"
    else:
        say(f"POST /learner/{learner_id}/intervene: HTTP {status} {package} (skipped)")

    trap = step("POST /reassess (trap, correct)", client.post("/reassess", {
        "learner_id": learner_id, "class": "M08", "item_id": "trap_M08", "item_type": "trap",
        "result": {"correct": True, "answer": "outside the array"}}))
    say(f"  state={trap['state']} p_active={trap['p_active']:.3f} next_item={trap['next_item']}")
    assert trap["state"] == "PROBATION"
    assert trap["next_item"]["item_type"] == "transfer_code"

    final = step("POST /reassess (transfer, passed)", client.post("/reassess", {
        "learner_id": learner_id, "class": "M08", "item_id": trap["next_item"]["item_id"],
        "item_type": "transfer_code", "result": {"passed": True}}))
    say(f"  state={final['state']} p_active={final['p_active']:.3f} resolved_level={final['resolved_level']}")
    for condition in final["conditions"]:
        say(f"    [{'x' if condition['met'] else ' '}] {condition['label']} ({condition['detail']})")
    assert final["state"] == "STABLE"
    assert all(c["met"] for c in final["conditions"])
    say("OK: seeded learner reassessed to STABLE")
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", help="base URL of a running server; default: in-process app with a temp database")
    args = parser.parse_args()
    if args.url:
        run(UrlClient(args.url))
        return
    os.environ.setdefault("RELEARN_NO_WARM", "1")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["RELEARN_DB"] = str(Path(tmp) / "demo.db")
        from fastapi.testclient import TestClient
        from server.app.main import app
        run(AppClient(TestClient(app)))


if __name__ == "__main__":
    main()
