# Setup the repo .venv via uv
setup:
    uv sync
    uv pip install --pre deepmimo

# Run static analysis and automatically fix issues where possible
check:
    uvx ruff check . --fix

# Format code according to project style
format:
    uvx ruff format .

# Run formatting and linting (CI-style target)
clean: format check

# Run simulations
sim := "uv run scripts/simulation.py"

fed:
    {{sim}} orchestrator=federated

ot:
    {{sim}} orchestrator=optimal_transport

fb:
    {{sim}} orchestrator=flat_bundle

cs:
    {{sim}} orchestrator=cover_sheaf

bun:
    {{sim}} orchestrator=bundle

vanilla:
    {{sim}} orchestrator=vanilla

sim-all: bun cs fb ot fed
