.PHONY: help setup backend frontend build dev test clean

# Use the venv that setup.sh creates; fall back to whatever python3 is around.
# abspath so targets that cd into backend/ still find it.
VENV_PY := $(firstword $(wildcard .venv/bin/python .venv/Scripts/python.exe))
PY := $(if $(VENV_PY),$(abspath $(VENV_PY)),python3)

help:
	@echo "ClipForge — make targets:"
	@echo "  setup     install backend + frontend deps"
	@echo "  backend   run the API (uvicorn) on :8000"
	@echo "  frontend  run the Vite dev server on :5173"
	@echo "  build     build the frontend into frontend/dist (served by backend at /)"
	@echo "  test      run backend unit tests"
	@echo "  clean     remove build artifacts and runtime data"

setup:
	./setup.sh

backend:
	cd backend && $(PY) -m uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

build:
	cd frontend && npm run build

test:
	cd backend && $(PY) -m tests.test_units

clean:
	rm -rf frontend/dist backend/data
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
