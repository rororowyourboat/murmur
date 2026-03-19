.PHONY: install install-all lint format format-check test check fix

install:
	uv sync --extra dev

install-all:
	uv sync --extra all

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

test:
	uv run pytest

check: lint format-check test

fix:
	uv run ruff check --fix .
