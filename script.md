# Demo Script — Re:Learn

Wiring check done: backend (`uvicorn server.app.main:app --port 8000`) and frontend (`npm run dev` in `web/`)
are already correctly wired on every route — `web/src/lib/api.ts` calls match the live routers in
`server/app/routes/*.py` exactly. Verified live with curl, not fixtures. No frontend changes were needed;
the two flows below are the ones to put in front of judges because they hit **real rule-engine diagnosis
over a custom C interpreter**, not canned fixture JSON.

Everything else (`/problems`, `/learner`, `/metrics`, `/quiz/*`, `/reassess`, `/code-items`, `/reason`) still
answers from `server/fixtures/*.json` — fine for scaffolding, but don't demo those as "the model."

---

## Feature 1 — Mission Flow: live misconception diagnosis (Conditions/Loops planets)

**Route:** `/planet/loops/mission/P01` → run code → `/planet/loops/diagnosis`
**Backend:** `POST /attempt` (`server/app/routes/attempt.py`) — real path, not fixture, for P01/P03/P05/P06
(loops), P11/P12/P16/P17 (conditions), and the arrays/variables/functions IDs.

### What to say
> "This isn't a static right/wrong checker. We wrote our own C interpreter from scratch
> (`ml/c_interp/`) that actually executes the student's code line-by-line, traces every loop
> iteration and variable mutation, then runs that trace through a Bayesian rule engine that
> names the *misconception* — not just that a test failed, but *why*, in plain English, with a
> confidence score."

### Live steps
1. Open `/planet/loops/mission/P01` ("fire_shots" — fire a cannon n times).
2. Paste this buggy solution into the editor (classic off-by-one):

```c
void fire_shots(int n) {
    for (int i = 0; i <= n; i++) {
        fire();
    }
}
```

3. Click **Run**. Tests fail (3 passed→0). Click **Continue** → lands on the diagnosis screen.
4. Point out the diagnosis panel: **"Boundary Drift" (M01), confidence 0.75**, evidence:
   *"Off by exactly one pass of the loop."*
5. Say: "That number comes from `server/app/conditions_rules.py`/`loops_rules.py` scoring the
   real execution trace against known misconception signatures — it's the same engine whether
   the code is right, wrong, or ambiguous between two lookalike bugs."

### Raw proof (paste in terminal if asked "is this really live?")
```bash
curl -s -X POST localhost:8000/attempt -H "Content-Type: application/json" -d '{
  "learner_id":"demo","problem_id":"P01",
  "code":"void fire_shots(int n) {\n    for (int i = 0; i <= n; i++) {\n        fire();\n    }\n}"
}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['diagnosis']['top'], d['diagnosis']['evidence'])"
```
Expected: `[{'id': 'M01', 'p': 0.75, 'name': 'Boundary Drift', ...}] [{'type': 'RUN', 'text': 'Off by exactly one pass of the loop.'}]`

---

## Feature 2 — Deep Space Trials: DSA misconception diagnosis under exam conditions

**Route:** `/trials/exam` (5-problem timed exam) → `/trials/debrief`
**Backend:** `POST /trials/start`, `POST /trials/run`, `POST /trials/answer`, `POST /trials/finish`
(`server/app/routes/trials.py`) — all live, session-tracked per `exam_id`, same real interpreter +
`server/app/dsa_rules.py` diagnoser underneath. (`/exam/*` is a separate, more advanced adaptive-exam
system from the ML pipeline merge — don't confuse the two when demoing.)

### What to say
> "Same engine, harder problems. This is a full timed exam UI — cockpit layout, trace
> visualization panel, predict-the-output questions — and every submission gets diagnosed
> against *known DSA misconceptions*, not just pass/fail. It also compiles a debrief report
> at the end naming exactly which bugs showed up and where to go practice them."

### Live steps
1. Open `/trials/exam` — it calls `/trials/start` and loads Trial 1 (`T3_bubble_sort` or
   `T1_two_sum`, whichever is first in rotation — use **Skip** to reach `T3_bubble_sort` if needed).
2. Paste this classic swap-without-a-temp bug into the editor:

```c
void bubble_sort(int a[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                a[j] = a[j + 1];
                a[j + 1] = a[j];
            }
        }
    }
}
```

3. Click **Run Samples**. 1/3 sample tests fail. The right-hand diagnostic panel surfaces the
   finding live (pulled straight from `res.diagnosis`).
4. Say: "The engine caught **Cargo Overwrite (D03) at 94% confidence** — it recognized the
   classic 'assign without saving to a temp' swap bug from the trace, named it, and explained
   it in one sentence, in real time, mid-exam."
5. Submit and finish the exam to reach `/trials/debrief` — the report compiles every
   misconception hit across all 5 trials into one scorecard.

### Raw proof
```bash
curl -s -X POST localhost:8000/trials/run -H "Content-Type: application/json" -d '{
  "item_id":"T3_bubble_sort",
  "code":"void bubble_sort(int a[], int n) {\n    for (int i = 0; i < n - 1; i++) {\n        for (int j = 0; j < n - 1 - i; j++) {\n            if (a[j] > a[j + 1]) {\n                a[j] = a[j + 1];\n                a[j + 1] = a[j];\n            }\n        }\n    }\n}"
}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['diagnosis']['top'], d['tests']['passed'],'/',d['tests']['total'])"
```
Expected: `[{'id': 'D03', 'p': 0.94, 'name': 'Cargo Overwrite', ...}] 1 / 3`

---

## If judges ask "what's fake / what's fixture"

Be upfront — it's a strength, not a weakness, to show you know the boundary:
- **Real (rule-engine + real C interpreter):** `/attempt`, `/run`, `/lab/diagnose`, `/intervene`,
  `/probe/answer` (P03 only) for Conditions/Loops/Arrays/Variables/Functions planets; the entire
  `/trials/*` family for Deep Space Trials.
- **Fixture-backed (scaffolding, same JSON shape the real routes return):** `/problems`,
  `/learner*`, `/metrics`, `/quiz/*`, `/reassess`, `/code-items*`, `/reason`, `/lab/reason`.
  Check `GET /_routes` on the running server any time to see the live split.

## Start commands
```bash
make serve          # backend on :8000
cd web && npm run dev  # frontend, picks the first free port from :5173
```
