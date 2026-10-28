.PHONY: setup test lint
setup: ; ./bootstrap.sh
test:  ; uv run pytest -q
lint:  ; uv run ruff check src tests
