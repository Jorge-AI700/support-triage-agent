.PHONY: up down ingest load-embeddings classify-sample classify-sample-v1 \
	classify-sample-v2 classify-test classify-test-v1 classify-test-v2 report \
	e2e e2e-v1 e2e-v2 e2e-report e2e-compare test api

up:
	docker compose up -d

down:
	docker compose down

ingest:
	python3 -m pipeline.ingest

load-embeddings:
	python3 scripts/load_embeddings.py

classify-sample: classify-sample-v2

classify-sample-v1:
	@run_id="$(if $(RUN),$(RUN),sample-v1-$$(date -u +%Y%m%dT%H%M%SZ))"; \
	python3 -m pipeline.classify --split test --limit 300 --run-id "$$run_id" --prompt-version v1

classify-sample-v2:
	@run_id="$(if $(RUN),$(RUN),sample-v2-$$(date -u +%Y%m%dT%H%M%SZ))"; \
	python3 -m pipeline.classify --split test --limit 300 --run-id "$$run_id" --prompt-version v2

classify-test: classify-test-v2

classify-test-v1:
	@run_id="$(if $(RUN),$(RUN),test-v1-$$(date -u +%Y%m%dT%H%M%SZ))"; \
	python3 -m pipeline.classify --split test --run-id "$$run_id" --prompt-version v1

classify-test-v2:
	@run_id="$(if $(RUN),$(RUN),test-v2-$$(date -u +%Y%m%dT%H%M%SZ))"; \
	python3 -m pipeline.classify --split test --run-id "$$run_id" --prompt-version v2

report:
	@test -n "$(RUN)" || (echo "Usage: make report RUN=<run-id>"; exit 1)
	python3 -m evals.classification_report --run-id "$(RUN)"

e2e: e2e-v2

e2e-v1:
	@run_id="$(if $(RUN),$(RUN),e2e-v1-$$(date -u +%Y%m%dT%H%M%SZ))"; \
	python3 -m pipeline.run_e2e --run-id "$$run_id" --limit "$(if $(N),$(N),200)" --prompt-version v1 $(if $(CLASSIFY_RUN),--classify-run-id "$(CLASSIFY_RUN)",)

e2e-v2:
	@run_id="$(if $(RUN),$(RUN),e2e-v2-$$(date -u +%Y%m%dT%H%M%SZ))"; \
	python3 -m pipeline.run_e2e --run-id "$$run_id" --limit "$(if $(N),$(N),200)" --prompt-version v2 $(if $(CLASSIFY_RUN),--classify-run-id "$(CLASSIFY_RUN)",)

e2e-report:
	@test -n "$(RUN)" || (echo "Usage: make e2e-report RUN=<run-id>"; exit 1)
	python3 -m evals.e2e_report --run-id "$(RUN)"

e2e-compare:
	@test -n "$(RUN_A)" -a -n "$(RUN_B)" || (echo "Usage: make e2e-compare RUN_A=<v1-run> RUN_B=<v2-run>"; exit 1)
	python3 -m evals.e2e_report --compare "$(RUN_A)" "$(RUN_B)"

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
