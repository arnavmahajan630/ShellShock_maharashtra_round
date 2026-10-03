# Endpoint status

Base URL: `http://<laptop address>:8000`. Interactive docs: `/docs`. Live list of what is real: `GET /_routes`.

**fixture** = answers with a fixed sample from `server/fixtures/`. The shape is final; the values are not computed.
**live** = computed by the real code. The owning package updates its row when it goes live.

| Endpoint | Status | Package | Fixture behaviour |
|---|---|---|---|
| `GET /health` | live | W0 | |
| `GET /problems` | fixture | S2 | 3 sample problems (P03, P11, Q17) |
| `GET /problems/{id}` | fixture | S2 | by id; unknown id → P03 |
| `POST /learner` | fixture | S1 | always `demo-learner` |
| `POST /learner/{id}/seed` | fixture | S1 | seeded demo learner |
| `GET /learner/{id}` | fixture | S1 | seeded demo learner |
| `POST /run` | fixture | S2 | by `problem_id`: P03, Q17 |
| `POST /attempt` | fixture | S2 | by `problem_id`: P03 → ambiguous M01/M08 with a probe; P11 → confident M07; Q06 → confident D03; Q17 → correct; `GATE` → gate G3b |
| `POST /probe/answer` | fixture | S2 | M08 at 0.90 after the probe |
| `POST /intervene` | fixture | S1 | M08 `memory_strip` package |
| `POST /reassess` | fixture | S1 | by `item_type`: `trap` → not yet; `transfer_code` → STABLE |
| `POST /lab/diagnose` | fixture | S2 | same cases as `/attempt` |
| `GET /metrics` | fixture | S3 | stub with `"stub": true` |
| `POST /exam/start` | fixture | S3 | 3-item exam, first item Q06 |
| `POST /exam/answer` | fixture | S3 | by `item_id`: Q06 → Q08 → xt_rec_1 → end |
| `POST /exam/finish` | fixture | S3 | report: D03 NEW, M01 HELD, D05 UNCERTAIN + 1 deferred probe |
| `POST /exam/probe` | fixture | S3 | same report with D05 sharpened to NEW |
| `GET /exam/{id}/report` | fixture | S3 | same as `/exam/finish` |
| `GET /quiz/next` | fixture | S3 | one collision item (M01 / M08) |
| `POST /quiz/answer` | fixture | S3 | by `answer`: `3` → wrong + `ask_reason: true`; `4` → correct |
| `GET /code-items` | fixture | S3 | one item of each type for P03 |
| `POST /code-items/debug` | fixture | S3 | by `line`: `4` → correct; anything else → wrong |
| `POST /reason` | fixture | S3 | matched M08 |
| `POST /lab/reason` | fixture | S3 | matched M08 |
