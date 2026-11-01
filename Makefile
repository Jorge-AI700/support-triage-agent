.PHONY: up down ingest load-embeddings classify-sample classify-test report e2e e2e-report test api

up:
	docker compose up -d

down:
	docker compose down

ingest:
	python3 -m pipeline.ingest

load-embeddings:
	python3 scripts/load_embeddings.py

classify-sample:
	@run_id="$(if $(RUN),$(RUN),sample-$$(date -u +%Y%m%dT%H%M%SZ))"; \
	python3 -m pipeline.classify --split test --limit 300 --run-id "$$run_id"

classify-test:
	@run_id="$(if $(RUN),$(RUN),test-$$(date -u +%Y%m%dT%H%M%SZ))"; \
	python3 -m pipeline.classify --split test --run-id "$$run_id"

report:
	@test -n "$(RUN)" || (echo "Usage: make report RUN=<run-id>"; exit 1)
	python3 -m evals.classification_report --run-id "$(RUN)"

e2e:
	@run_id="$(if $(RUN),$(RUN),e2e-$$(date -u +%Y%m%dT%H%M%SZ))"; \
	python3 -m pipeline.run_e2e --run-id "$$run_id" --limit "$(if $(N),$(N),200)" $(if $(CLASSIFY_RUN),--classify-run-id "$(CLASSIFY_RUN)",)

e2e-report:
	@test -n "$(RUN)" || (echo "Usage: make e2e-report RUN=<run-id>"; exit 1)
	python3 -m evals.e2e_report --run-id "$(RUN)"

test:
	@if command -v pytest >/dev/null 2>&1; then \
		pytest evals/; \
	elif [ -x .venv/bin/pytest ]; then \
		.venv/bin/pytest evals/; \
	else \
		.venv/bin/python3 -m evals.test_split 2>/dev/null || python3 -m evals.test_split; \
	fi

api:
	uvicorn app.main:app --reload --port 8000
