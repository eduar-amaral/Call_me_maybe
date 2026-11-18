install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

lint:
	uv run flake8 .
	uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

clean:
	find . -type d \( \
		-name "__pycache__" -o \
		-name ".mypy_cache" -o \
		-name ".pytest_cache" -o \
		-name ".ruff_cache" -o \
		-name ".tox" \
	\) -prune -exec rm -rf {} +
	find . -type f \( \
		-name "*.pyc" -o \
		-name "*.pyo" \
	\) -delete
	rm -rf .coverage coverage.xml htmlcov

fclean: clean
	rm -rf .venv

.PHONY: install run debug lint lint-strict clean fclean
