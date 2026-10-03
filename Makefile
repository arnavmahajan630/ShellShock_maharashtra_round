# Run with `mingw32-make <target>` on Windows (or `make` if installed).
PY := .venv/Scripts/python

.PHONY: setup test fixtures serve data eda train eval demo

setup:
	python -m venv .venv
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests -q

# Rebuild the sample data and the fake API responses (package W0).
fixtures:
	$(PY) -m tests.fixtures.build

serve:
	$(PY) -m uvicorn server.app.main:app --host 0.0.0.0 --port 8000

# The targets below call modules that later packages provide (plans/06).
data:
	$(PY) -m ml.generate.build_dataset

eda:
	$(PY) -m ml.eda.eda

train:
	$(PY) -m ml.model.train

eval:
	$(PY) -m ml.eval.run_all

demo: serve
