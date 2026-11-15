test:
	uv run pytest --cov

ruff:
	-uv run ruff format
	@uv run ruff check --fix

mypy:
	uv run mypy lapeft_bayesopt

lint:
	make ruff
	make mypy

build:
	uv build
