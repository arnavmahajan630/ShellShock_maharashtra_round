"""SQLite store for learners (package S1): learners, attempts, knowledge, events.

No FastAPI in here. `routes/learner.py` is the HTTP layer; S2, S3 and I1 can call these
functions directly (for example `store.get_store().apply_attempt_result(...)`).

Where the database lives
    $RELEARN_DB if set, else server/data/relearn.db (`*.db` is git-ignored). Each call of
    `get_store()` looks at the variable again, so a test that points it at a temp file gets
    a fresh store. Use ":memory:" only for a single store object you keep yourself.

Tables (03 11.3): learners, attempts, knowledge, events.
    knowledge(learner_id, class, p_active, state, times_seen, last_tested, recheck_queued, log_json)
    `log_json` = {"evidence": [...], "interventions": [...], "flags": {...}}. `flags` is private
    bookkeeping that is not part of the public KnowledgeEntry: the probation checklist
    (trap, transfer), items already asked in this probation, the family where the
    misconception was found.

All probability work is `ml.learner.knowledge.update_p`; all transitions are
`ml.learner.state_machine.next_state`. Nothing in here invents a number.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ml.contracts.classes import CLASS_INFO, MISCONCEPTIONS
from ml.contracts.params import (DIAGNOSIS_ACTIVE_P, ITEM_GUESS_SLIP, POPULATION_PRIOR)
from ml.learner import knowledge, state_machine

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "server" / "data" / "relearn.db"
ITEMS_PATH = ROOT / "ml" / "data" / "items.json"

SCHEMA = """
CREATE TABLE IF NOT EXISTS learners (
    learner_id TEXT PRIMARY KEY,
    callsign   TEXT,
    created_at TEXT NOT NULL,
    nodes_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    problem_id TEXT NOT NULL,
    ts         TEXT NOT NULL,
    passed     INTEGER NOT NULL,
    total      INTEGER NOT NULL,
    status     TEXT NOT NULL,
    top        TEXT,
    code       TEXT
);
CREATE INDEX IF NOT EXISTS attempts_by_learner ON attempts (learner_id, seq);
CREATE TABLE IF NOT EXISTS knowledge (
    learner_id     TEXT NOT NULL,
    class          TEXT NOT NULL,
    p_active       REAL NOT NULL,
    state          TEXT NOT NULL,
    times_seen     INTEGER NOT NULL DEFAULT 0,
    last_tested    TEXT,
    recheck_queued INTEGER NOT NULL DEFAULT 0,
    log_json       TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (learner_id, class)
);
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT NOT NULL,
    ts         TEXT NOT NULL,
    type       TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS events_by_learner ON events (learner_id, id);
"""

MAX_EVIDENCE = 20          # evidence items kept per (learner, class), most recent last

# item_type of /reassess -> the knowledge.update_p kind. The request names the item; the
# update row (guess/slip) is decided here from 03 8.2.
REASSESS_KINDS = {
    "trap": "trap",
    "transfer_code": "transfer_code",
    "same_family_code": "same_family_code",
    "ghost": "ghost",
    "exam_code": "exam_code",
    "exam_trace": "exam_trace",
    "belief_mcq": "belief_mcq",
    "mcq": "mcq",
    "probe": "belief_mcq",
    "predict_output": "predict_output",
    "next_state": "next_state",
}
TRAP_TYPES = ("trap", "predict_output")      # a predict_output item whose belief names the class is a trap
COMPLETION_TYPES = ("intervention_done", "intervention_complete")
PROBATION_TYPES = ("trap", "transfer_code", "same_family_code", "predict_output")   # probation checklist items


class StoreError(ValueError):
    """The request is wrong (unknown learner, unknown class, bad item type). The route turns it into a 4xx."""

    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def db_path():
    configured = os.environ.get("RELEARN_DB")
    return configured if configured else str(DEFAULT_DB)


_stores = {}
_stores_lock = threading.Lock()


def get_store():
    """The store for the current $RELEARN_DB (created and cached per path)."""
    path = db_path()
    with _stores_lock:
        if path not in _stores:
            _stores[path] = Store(path)
        return _stores[path]


# ---------------------------------------------------------------- the demo learner (03 11: seed)

SEED_CALLSIGN = "NOVA"

# (attempt_id, problem_id, ts, passed, total, status, top): ~12 attempts over three planets.
SEED_ATTEMPTS = [
    ("at_001", "P11", "2026-10-04T10:00:00Z", 5, 5, "correct", "CORRECT"),
    ("at_002", "P01", "2026-10-04T10:06:00Z", 0, 4, "confident", "M01"),
    ("at_003", "P03", "2026-10-04T10:15:00Z", 0, 5, "ambiguous", "M01"),
    ("at_004", "P12", "2026-10-04T10:21:00Z", 3, 5, "confident", "M06"),
    ("at_005", "P12", "2026-10-04T10:26:00Z", 5, 5, "correct", "CORRECT"),
    ("at_006", "P01", "2026-10-04T10:31:00Z", 4, 4, "correct", "CORRECT"),
    ("at_007", "P16", "2026-10-04T10:38:00Z", 5, 5, "correct", "CORRECT"),
    ("at_008", "P17", "2026-10-04T10:44:00Z", 6, 6, "correct", "CORRECT"),
    ("at_009", "P03", "2026-10-04T10:52:00Z", 1, 5, "confident", "M08"),
    ("at_010", "P08", "2026-10-04T11:00:00Z", 5, 5, "correct", "CORRECT"),
    ("at_011", "P13", "2026-10-04T11:05:00Z", 4, 4, "correct", "CORRECT"),
    ("at_012", "P03", "2026-10-04T11:12:00Z", 2, 5, "confident", "M08"),
]

SEED_NODES = {
    "variables:P13": {"status": "done", "stars": 3},
    "conditions:P11": {"status": "done", "stars": 3},
    "conditions:P12": {"status": "done", "stars": 2},
    "conditions:P16": {"status": "done", "stars": 3},
    "conditions:P17": {"status": "done", "stars": 3},
    "loops:P01": {"status": "done", "stars": 2},
    "loops:P03": {"status": "unstable", "stars": 0},
    "arrays:P08": {"status": "done", "stars": 3},
}

# class -> (state, p_active, times_seen, evidence, interventions, last_tested, recheck, flags)
SEED_KNOWLEDGE = {
    "M01": ("STABLE", 0.08, 2, [{"type": "RUN", "text": "Loop ran 6 times; the mission needed 5."}],
            ["trace_timeline"], "loops:P01", True,
            {"found_family": "count_loop", "trap_passed": True, "trap_detail": "answered: 5",
             "transfer_passed": True, "transfer_detail": "P02 countdown", "asked": ["trap_M01", "P02"]}),
    "M06": ("MASTERED", 0.04, 2, [{"type": "RUN", "text": "Ghost return P17 passed."}],
            ["trace_timeline"], "conditions:P17", False,
            {"found_family": "equality_check", "trap_passed": True, "trap_detail": "answered: door opens",
             "transfer_passed": True, "transfer_detail": "P17 max_of_three", "asked": ["trap_M06", "P17"]}),
    "M08": ("ACTIVE", 0.9, 1, [{"type": "PROBE", "text": "You said the last valid cell of `int a[5]` is 5."}],
            [], "loops:P03", False, {"found_family": "array_accumulate"}),
}


def _blank_flags():
    return {"found_family": None, "trap_passed": False, "trap_detail": "", "transfer_passed": False,
            "transfer_detail": "", "asked": []}


def _blank_entry():
    return {"state": "UNSEEN", "p_active": POPULATION_PRIOR, "times_seen": 0, "evidence": [], "interventions": [],
            "last_tested": None, "recheck_queued": False, "flags": _blank_flags()}


class Store:
    """One SQLite file. A new connection per call, one process-wide lock around read-modify-write."""

    def __init__(self, path):
        self.path = str(path)
        self._lock = threading.RLock()
        self._memory = None
        if self.path == ":memory:":
            self._memory = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(SCHEMA)
                conn.commit()
            finally:
                self._release(conn)

    # ------------------------------------------------------------ plumbing

    def _connect(self):
        if self._memory is not None:
            self._memory.row_factory = sqlite3.Row
            return self._memory
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _release(self, conn):
        if conn is not self._memory:
            conn.close()

    def _run(self, fn):
        """Run fn(conn) under the lock in one transaction."""
        with self._lock:
            conn = self._connect()
            try:
                result = fn(conn)
                conn.commit()
                return result
            except Exception:
                conn.rollback()
                raise
            finally:
                self._release(conn)

    # ------------------------------------------------------------ learners

    def exists(self, learner_id):
        return self._run(lambda c: c.execute("SELECT 1 FROM learners WHERE learner_id=?", (learner_id,)).fetchone()
                         is not None)

    def create_learner(self, callsign=None, learner_id=None):
        """New learner with all 17 classes UNSEEN at the population prior. Returns the id."""
        learner_id = learner_id or f"lrn_{uuid.uuid4().hex[:10]}"

        def work(conn):
            if conn.execute("SELECT 1 FROM learners WHERE learner_id=?", (learner_id,)).fetchone():
                raise StoreError(f"learner {learner_id} already exists", 409)
            self._insert_blank(conn, learner_id, callsign)
        self._run(work)
        return learner_id

    def _insert_blank(self, conn, learner_id, callsign):
        conn.execute("INSERT INTO learners (learner_id, callsign, created_at, nodes_json) VALUES (?,?,?,?)",
                     (learner_id, callsign, _now(), "{}"))
        for cls in MISCONCEPTIONS:
            self._write_entry(conn, learner_id, cls, _blank_entry())

    def seed(self, learner_id, callsign=SEED_CALLSIGN):
        """Reset `learner_id` to the demo learner (created if it does not exist). Deterministic."""
        def work(conn):
            row = conn.execute("SELECT callsign FROM learners WHERE learner_id=?", (learner_id,)).fetchone()
            keep = row["callsign"] if row and row["callsign"] else callsign
            for table in ("attempts", "knowledge", "events", "learners"):
                conn.execute(f"DELETE FROM {table} WHERE learner_id=?", (learner_id,))
            self._insert_blank(conn, learner_id, keep)
            conn.execute("UPDATE learners SET nodes_json=? WHERE learner_id=?",
                         (json.dumps(SEED_NODES), learner_id))
            for seq, (aid, pid, ts, passed, total, status, top) in enumerate(SEED_ATTEMPTS, start=1):
                conn.execute("INSERT INTO attempts (attempt_id, learner_id, seq, problem_id, ts, passed, total, status,"
                             " top, code) VALUES (?,?,?,?,?,?,?,?,?,NULL)",
                             (f"{learner_id}:{aid}" if learner_id != "demo-learner" else aid, learner_id, seq, pid, ts,
                              passed, total, status, top))
            for cls, (state, p, seen, evidence, interventions, last, recheck, flags) in SEED_KNOWLEDGE.items():
                entry = {"state": state, "p_active": p, "times_seen": seen, "evidence": list(evidence),
                         "interventions": list(interventions), "last_tested": last, "recheck_queued": recheck,
                         "flags": {**_blank_flags(), **flags}}
                self._write_entry(conn, learner_id, cls, entry)
            self._event(conn, learner_id, "seed", {"callsign": keep})
        self._run(work)
        return self.get_learner(learner_id)

    def get_learner(self, learner_id):
        """The `Learner` object of 03 11 (schemas.Learner)."""
        def work(conn):
            row = conn.execute("SELECT * FROM learners WHERE learner_id=?", (learner_id,)).fetchone()
            if row is None:
                raise StoreError(f"unknown learner {learner_id}", 404)
            misconceptions = {}
            for krow in conn.execute("SELECT * FROM knowledge WHERE learner_id=?", (learner_id,)):
                misconceptions[krow["class"]] = _public_entry(_entry_from_row(krow))
            ordered = {cls: misconceptions[cls] for cls in MISCONCEPTIONS if cls in misconceptions}
            attempts = [{"attempt_id": a["attempt_id"], "problem_id": a["problem_id"], "ts": a["ts"],
                         "passed": a["passed"], "total": a["total"], "status": a["status"], "top": a["top"]}
                        for a in conn.execute("SELECT * FROM attempts WHERE learner_id=? ORDER BY seq", (learner_id,))]
            return {"learner_id": learner_id, "callsign": row["callsign"], "misconceptions": ordered,
                    "nodes": json.loads(row["nodes_json"]), "attempts": attempts}
        return self._run(work)

    def prior(self, learner_id):
        """{class: P(active)} for the Bayes layer, or None when the learner has no history yet."""
        try:
            learner = self.get_learner(learner_id)
        except StoreError:
            return None
        entries = learner["misconceptions"]
        if not any(e["state"] != "UNSEEN" or e["times_seen"] for e in entries.values()):
            return None
        return {cls: e["p_active"] for cls, e in entries.items()}

    # ------------------------------------------------------------ knowledge rows

    def _write_entry(self, conn, learner_id, cls, entry):
        log = {"evidence": entry["evidence"][-MAX_EVIDENCE:], "interventions": entry["interventions"],
               "flags": entry["flags"]}
        conn.execute(
            "INSERT OR REPLACE INTO knowledge (learner_id, class, p_active, state, times_seen, last_tested,"
            " recheck_queued, log_json) VALUES (?,?,?,?,?,?,?,?)",
            (learner_id, cls, float(entry["p_active"]), entry["state"], int(entry["times_seen"]), entry["last_tested"],
             int(bool(entry["recheck_queued"])), json.dumps(log)))

    def _read_entry(self, conn, learner_id, cls):
        if cls not in MISCONCEPTIONS:
            raise StoreError(f"unknown class {cls!r}")
        if conn.execute("SELECT 1 FROM learners WHERE learner_id=?", (learner_id,)).fetchone() is None:
            raise StoreError(f"unknown learner {learner_id}", 404)
        row = conn.execute("SELECT * FROM knowledge WHERE learner_id=? AND class=?", (learner_id, cls)).fetchone()
        return _entry_from_row(row) if row is not None else _blank_entry()

    def get_entry(self, learner_id, cls):
        """Full entry including private `flags`."""
        return self._run(lambda c: self._read_entry(c, learner_id, cls))

    def _event(self, conn, learner_id, type_, payload):
        conn.execute("INSERT INTO events (learner_id, ts, type, payload_json) VALUES (?,?,?,?)",
                     (learner_id, _now(), type_, json.dumps(payload, default=str)))

    def log_event(self, learner_id, type_, payload=None):
        self._run(lambda c: self._event(c, learner_id, type_, payload or {}))

    def events(self, learner_id, type_=None):
        def work(conn):
            sql, args = "SELECT * FROM events WHERE learner_id=?", [learner_id]
            if type_:
                sql, args = sql + " AND type=?", args + [type_]
            return [{"id": r["id"], "ts": r["ts"], "type": r["type"], "payload": json.loads(r["payload_json"])}
                    for r in conn.execute(sql + " ORDER BY id", args)]
        return self._run(work)

    # ------------------------------------------------------------ attempts

    def record_attempt(self, learner_id, attempt_id, problem_id, passed, total, status, top=None, code=None, ts=None):
        def work(conn):
            if conn.execute("SELECT 1 FROM learners WHERE learner_id=?", (learner_id,)).fetchone() is None:
                raise StoreError(f"unknown learner {learner_id}", 404)
            seq = conn.execute("SELECT COALESCE(MAX(seq),0)+1 FROM attempts WHERE learner_id=?",
                               (learner_id,)).fetchone()[0]
            conn.execute("INSERT OR REPLACE INTO attempts (attempt_id, learner_id, seq, problem_id, ts, passed, total,"
                         " status, top, code) VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (attempt_id, learner_id, seq, problem_id, ts or _now(), int(passed), int(total), status, top,
                          code))
        self._run(work)

    def get_attempt(self, attempt_id, learner_id=None):
        def work(conn):
            sql, args = "SELECT * FROM attempts WHERE attempt_id=?", [attempt_id]
            if learner_id:
                sql, args = sql + " AND learner_id=?", args + [learner_id]
            row = conn.execute(sql, args).fetchone()
            return dict(row) if row else None
        return self._run(work)

    def set_node(self, learner_id, node_id, status, stars):
        def work(conn):
            row = conn.execute("SELECT nodes_json FROM learners WHERE learner_id=?", (learner_id,)).fetchone()
            if row is None:
                raise StoreError(f"unknown learner {learner_id}", 404)
            nodes = json.loads(row["nodes_json"])
            nodes[node_id] = {"status": status, "stars": int(stars)}
            conn.execute("UPDATE learners SET nodes_json=? WHERE learner_id=?", (json.dumps(nodes), learner_id))
        self._run(work)

    def apply_attempt_result(self, learner_id, attempt_id, problem_id, passed, total, status, top=None, top_p=None,
                             exposure=None, latent=None, family=None, code=None, node=None):
        """Record one /attempt outcome and apply the 03 8.2 code-task update. Returns {"updates": [...]}.

        failed task, diagnosis top-1 = k (a misconception, status confident or two_bug, top_p >= 0.5):
            signature failure for k with the problem's exposure `exposure` (e_ik; default 0.5).
        passed task with `latent` = k:  the same ratio with e_ik x 0.5.
        Anything else (ambiguous, novel, correct, gate) changes no P; the attempt is still logged.
        """
        updates = []

        def work(conn):
            if conn.execute("SELECT 1 FROM learners WHERE learner_id=?", (learner_id,)).fetchone() is None:
                raise StoreError(f"unknown learner {learner_id}", 404)
            seq = conn.execute("SELECT COALESCE(MAX(seq),0)+1 FROM attempts WHERE learner_id=?",
                               (learner_id,)).fetchone()[0]
            conn.execute("INSERT OR REPLACE INTO attempts (attempt_id, learner_id, seq, problem_id, ts, passed, total,"
                         " status, top, code) VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (attempt_id, learner_id, seq, problem_id, _now(), int(passed), int(total), status, top, code))
            all_passed = total > 0 and passed == total
            target = kind = None
            if (not all_passed and top in MISCONCEPTIONS and status in ("confident", "two_bug")
                    and (top_p is None or float(top_p) >= DIAGNOSIS_ACTIVE_P)):
                target, kind = top, "signature_failure"
            elif all_passed and latent in MISCONCEPTIONS:
                target, kind = latent, "latent_pass"
            if target is not None:
                entry = self._read_entry(conn, learner_id, target)
                before = entry["p_active"]
                entry["p_active"] = knowledge.update_p(before, kind, exposure=exposure)
                entry["times_seen"] += 1
                entry["last_tested"] = node or problem_id
                if family and not entry["flags"].get("found_family") and entry["state"] in ("UNSEEN", "ACTIVE"):
                    entry["flags"]["found_family"] = family
                entry["evidence"].append({"type": "RUN", "text": f"{problem_id}: {passed}/{total} tests passed, "
                                          f"diagnosed {target} ({kind.replace('_', ' ')})."})
                new_state = state_machine.next_state(entry["state"], entry["p_active"],
                                                     {"after_item": True, "ghost_failed": False})
                entry["state"] = new_state
                self._write_entry(conn, learner_id, target, entry)
                updates.append({"class": target, "p_before": before, "p_after": entry["p_active"],
                                "state": new_state, "kind": kind})
            if node:
                row = conn.execute("SELECT nodes_json FROM learners WHERE learner_id=?", (learner_id,)).fetchone()
                nodes = json.loads(row["nodes_json"])
                stars = nodes.get(node, {}).get("stars", 0)
                nodes[node] = {"status": "done" if all_passed else "unstable",
                               "stars": max(stars, 3 if all_passed else 0)}
                conn.execute("UPDATE learners SET nodes_json=? WHERE learner_id=?", (json.dumps(nodes), learner_id))
            self._event(conn, learner_id, "attempt", {"attempt_id": attempt_id, "problem_id": problem_id,
                                                      "status": status, "updates": updates})
        self._run(work)
        return {"updates": updates}

    # ------------------------------------------------------------ interventions

    def tried_modalities(self, learner_id, cls):
        return list(self.get_entry(learner_id, cls)["interventions"])

    def start_intervention(self, learner_id, cls, modality, problem_id=None, family=None):
        """An intervention package was served. ACTIVE or RELAPSED -> TREATING; the modality is remembered.

        UNSEEN, PROBATION, STABLE and MASTERED keep their state (an intervention that nobody asked
        for by diagnosis, or a second look, is not a state change).
        """
        def work(conn):
            entry = self._read_entry(conn, learner_id, cls)
            before = entry["state"]
            entry["state"] = state_machine.next_state(before, entry["p_active"], {"intervention": "start"})
            if modality and modality not in entry["interventions"]:
                entry["interventions"].append(modality)
            if family and not entry["flags"].get("found_family"):
                entry["flags"]["found_family"] = family
            entry["last_tested"] = problem_id or entry["last_tested"]
            self._write_entry(conn, learner_id, cls, entry)
            self._event(conn, learner_id, "intervene", {"class": cls, "modality": modality, "from": before,
                                                        "to": entry["state"], "problem_id": problem_id})
            return _public_entry(entry)
        return self._run(work)

    # ------------------------------------------------------------ reassess (03 8.2, 8.4)

    def reassess(self, learner_id, cls, item_id, item_type, result, problem_lookup=None, problem_bank=None):
        """Apply one item response and return the /reassess body (without envelope fields).

        `result`: {correct|passed: bool, hint?: bool, answer?, correct_answer?, family?, problem_id?, name?,
                   belief?: {class: answer}, intervening_levels?: int}.
        `problem_lookup(problem_id)` -> full problem dict, used for the transfer-family check.
        `problem_bank`: list of full problem dicts, used to choose `next_item`.
        item_type "intervention_done" only finishes the intervention (TREATING -> PROBATION).
        """
        if item_type not in REASSESS_KINDS and item_type not in COMPLETION_TYPES:
            raise StoreError(f"unknown item_type {item_type!r}")
        result = dict(result or {})
        problem_lookup = problem_lookup or (lambda _pid: None)

        def work(conn):
            entry = self._read_entry(conn, learner_id, cls)
            flags = entry["flags"]
            state_before, p_before = entry["state"], entry["p_active"]

            if entry["state"] in ("ACTIVE", "RELAPSED") and (item_type in PROBATION_TYPES
                                                             or item_type in COMPLETION_TYPES):
                # The learner got to a reassessment item without a recorded intervention (the stateless
                # /intervene answered, or the app skipped the call): treat it as started and finished.
                entry["state"] = state_machine.next_state(entry["state"], entry["p_active"], {"intervention": "start"})
                entry["evidence"].append({"type": "RUN", "text": "Intervention taken as completed (not recorded)."})
            if entry["state"] == "TREATING":
                # A reassessment item (or an explicit completion) means the intervention is over.
                entry["p_active"] = knowledge.update_p(entry["p_active"], "intervention")
                entry["state"] = state_machine.next_state("TREATING", entry["p_active"], {"intervention": "done"})
                flags.update(trap_passed=False, trap_detail="", transfer_passed=False, transfer_detail="",
                             asked=[])
            if item_type in COMPLETION_TYPES:
                self._finish(conn, learner_id, cls, entry, state_before, p_before, item_id, item_type, result)
                return self._reassess_body(cls, entry, problem_bank)

            if "correct" in result:
                correct = bool(result["correct"])
            elif "passed" in result:
                correct = bool(result["passed"])
            else:
                raise StoreError("result needs `correct` (or `passed`)")
            hint = bool(result.get("hint", False))

            kind = REASSESS_KINDS[item_type]
            if kind == "ghost" and entry["state"] in ("STABLE", "MASTERED"):
                entry["p_active"] = knowledge.update_p(entry["p_active"], "forgetting",
                                                       levels=int(result.get("intervening_levels", 0) or 0))
            is_trap = kind == "trap"
            if kind == "predict_output":
                kind = knowledge.resolve_kind("predict_output", {"belief": result.get("belief") or {}, "class_id": cls})
                is_trap = kind == "predict_output_trap"
            entry["p_active"] = knowledge.update_p(entry["p_active"], kind, correct=correct, hint=hint)
            entry["times_seen"] += 1
            entry["last_tested"] = result.get("problem_id") or item_id
            if item_id not in flags["asked"]:
                flags["asked"].append(item_id)

            # Probation checklist (03 8.4): the trap and a different-family transfer, each only when passed.
            transfer_ok = None
            if is_trap:
                if correct:
                    flags["trap_passed"] = True
                flags["trap_detail"] = self._trap_detail(item_id, correct, result)
            elif item_type == "transfer_code":
                family = self._family(result, item_id, problem_lookup)
                transfer_ok = family is None or state_machine.is_transfer(cls, flags.get("found_family"), family)
                if correct and transfer_ok:
                    flags["transfer_passed"] = True
                    flags["transfer_detail"] = self._transfer_detail(item_id, result, problem_lookup)
                elif correct:
                    flags["transfer_detail"] = f"{item_id}: same family as where it was found, does not count"
                else:
                    flags["transfer_detail"] = f"{item_id} failed"

            event = {"after_item": True, "trap_passed": flags["trap_passed"],
                     "transfer_passed": flags["transfer_passed"]}
            if kind == "ghost":
                event.update(ghost_passed=correct, ghost_failed=not correct)
            if kind == "exam_code" or kind == "exam_trace":
                event.update(exam_passed=correct, exposure=result.get("exposure"))
            entry["state"] = state_machine.next_state(entry["state"], entry["p_active"], event)
            entry["evidence"].append({"type": "RUN", "text": f"{item_type} {item_id}: "
                                      f"{'passed' if correct else 'failed'}."})
            self._finish(conn, learner_id, cls, entry, state_before, p_before, item_id, item_type, result)
            return self._reassess_body(cls, entry, problem_bank)

        return self._run(work)

    def _finish(self, conn, learner_id, cls, entry, state_before, p_before, item_id, item_type, result):
        self._write_entry(conn, learner_id, cls, entry)
        self._event(conn, learner_id, "reassess", {"class": cls, "item_id": item_id, "item_type": item_type,
                                                   "result": result, "state_before": state_before,
                                                   "state_after": entry["state"], "p_before": p_before,
                                                   "p_after": entry["p_active"]})

    def _reassess_body(self, cls, entry, problem_bank):
        flags = entry["flags"]
        conditions = state_machine.reassess_conditions(
            entry["p_active"], trap_passed=flags["trap_passed"], transfer_passed=flags["transfer_passed"],
            trap_detail=flags["trap_detail"] or "not attempted yet",
            transfer_detail=flags["transfer_detail"] or "not attempted yet")
        return {"state": entry["state"], "p_active": round(entry["p_active"], 6), "conditions": conditions,
                "resolved_level": entry["state"] if entry["state"] in ("STABLE", "MASTERED") else None,
                "next_item": choose_next_item(cls, entry, problem_bank)}

    @staticmethod
    def _family(result, item_id, problem_lookup):
        family = result.get("family")
        if family:
            return family
        problem = problem_lookup(result.get("problem_id") or item_id)
        return problem.get("family") if problem else None

    @staticmethod
    def _transfer_detail(item_id, result, problem_lookup):
        problem = problem_lookup(result.get("problem_id") or item_id)
        name = result.get("name") or (problem.get("name") if problem else None)
        pid = result.get("problem_id") or item_id
        return f"{pid} {str(name).lower().replace(' ', '_')}" if name else str(pid)

    @staticmethod
    def _trap_detail(item_id, correct, result):
        answer = result.get("answer")
        right = result.get("correct_answer")
        if right is None:
            item = trap_item(item_id)
            right = item["correct"] if item else None
        if correct:
            return f"answered: {answer if answer is not None else right}" if (answer is not None or right) else "passed"
        if answer is not None and right is not None:
            return f"predicted {answer}, actual: {right}"
        return "failed"

    def next_item(self, learner_id, cls, problem_bank=None):
        """The item to offer next for `cls` (03 8.4) or None. {"item_id", "item_type"}."""
        entry = self.get_entry(learner_id, cls)
        return choose_next_item(cls, entry, problem_bank)


# ---------------------------------------------------------------- helpers

def _entry_from_row(row):
    log = json.loads(row["log_json"] or "{}")
    return {"state": row["state"], "p_active": row["p_active"], "times_seen": row["times_seen"],
            "evidence": log.get("evidence", []), "interventions": log.get("interventions", []),
            "last_tested": row["last_tested"], "recheck_queued": bool(row["recheck_queued"]),
            "flags": {**_blank_flags(), **log.get("flags", {})}}


def _public_entry(entry):
    """KnowledgeEntry fields only (schemas.KnowledgeEntry has extra='forbid')."""
    return {"state": entry["state"], "p_active": round(entry["p_active"], 6), "times_seen": entry["times_seen"],
            "evidence": list(entry["evidence"]), "interventions": list(entry["interventions"]),
            "last_tested": entry["last_tested"], "recheck_queued": bool(entry["recheck_queued"])}


_items_cache = {}


def trap_items():
    """ml/data/items.json by item_id (one trap per class)."""
    mtime = ITEMS_PATH.stat().st_mtime if ITEMS_PATH.exists() else None
    if _items_cache.get("mtime") != mtime:
        items = json.loads(ITEMS_PATH.read_text(encoding="utf-8")) if mtime else []
        _items_cache.update(mtime=mtime, items={i["item_id"]: i for i in items})
    return _items_cache["items"]


def trap_item(item_id):
    return trap_items().get(item_id)


def choose_next_item(cls, entry, problem_bank=None):
    """Next /reassess item. Only a class on PROBATION has one.

    Order: the class's trap (it carries the weight, 03 8.4) unless it was already asked in this
    probation, then a transfer problem from another family of the 03 8.3 table that was not asked
    yet. `problem_bank`: iterable of full problem dicts. A failed trap is not offered again; the
    second trap pool (predict_output quiz items) is S3's.
    """
    if entry["state"] != "PROBATION":
        return None
    flags = entry["flags"]
    asked = set(flags.get("asked", []))
    trap_id = f"trap_{cls}"
    if not flags.get("trap_passed") and trap_id not in asked and trap_item(trap_id):
        return {"item_id": trap_id, "item_type": "trap"}
    if not flags.get("transfer_passed"):
        found = flags.get("found_family")
        for problem in sorted(problem_bank or [], key=lambda p: p["problem_id"]):
            if problem["problem_id"] in asked:
                continue
            if state_machine.is_transfer(cls, found, problem.get("family")):
                return {"item_id": problem["problem_id"], "item_type": "transfer_code"}
    return None


def class_name(cls):
    return CLASS_INFO.get(cls, {}).get("name", cls)
