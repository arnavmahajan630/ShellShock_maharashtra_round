# The plans moved

This folder is kept only as a signpost. The plan files now live in two folders at the repo root:

| Was | Is now |
|---|---|
| `plans/01_SCOPE_v3.md` | `design_plan/01_SCOPE_v3.md` |
| `plans/02_DESIGN_SCOPE_v3.md` | `design_plan/02_DESIGN_SCOPE_v3.md` |
| (new) | `design_plan/05_CORE_DESIGN_SCOPE.md` |
| `plans/03_ML_IMPLEMENTATION_PLAN_v3.md` | `ml_plan/03_ML_IMPLEMENTATION_PLAN_v3.md` |
| `plans/04_FEASIBILITY_AND_ML_BUILD_PLAN.md` | `ml_plan/04_FEASIBILITY_AND_ML_BUILD_PLAN.md` |
| `plans/05_QUESTION_TYPES_AND_SENTENCE_READER.md` | `ml_plan/05_QUESTION_TYPES_AND_SENTENCE_READER.md` |
| `plans/06_AGENT_WORK_PACKAGES.md` | `ml_plan/06_AGENT_WORK_PACKAGES.md` |

A comment or note that still says `plans/03 §2.1` means `ml_plan/03_…`, section 2.1. Numbers 01 and 02 are in `design_plan/`; 03 to 06 are in `ml_plan/`.

Other places to look:

| Folder | What is in it |
|---|---|
| `ml/` | interpreter, features, model, item banks, sentence reader |
| `server/` | the API; `server/READY.md` lists every endpoint |
| `web/` | the web app |
| `ui/` | design images |
| `notes/` | one file per work package: what was built, decided and left open |
| `tests/` | one folder per work package |
