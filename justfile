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

# Run wandb leet for experiment tracking
leet:
    # Run wandb leet for experiment tracking
    uv run wandb beta leet run wandb

# Run simulation with given orchestrator (plus optional extra args)
sim orchestrator *args:
    uv run scripts/simulation.py orchestrator={{orchestrator}} {{args}}

# Federated learning orchestrator
fed:
    # Federated Channel Charting - standard federated learning approach
    just sim federated

# Optimal Transport orchestrator
ot:
    # Optimal Transport Channel Charting - uses optimal transport for aggregation
    just sim optimal_transport

# Flat Bundle orchestrator
fb:
    # Flat Bundle Channel Charting - bundle-based approach without sheaf structure
    just sim flat_bundle

# Cover Sheaf orchestrator
cs:
    # Cover Sheaf Channel Charting - sheaf-based with cover decomposition
    just sim cover_sheaf

# Bundle orchestrator
bun:
    # Bundle Channel Charting - bundle-based approach
    just sim bundle

# Diagonal Sheaf orchestrator
ds:
    # Diagonal Sheaf Channel Charting - sheaf with diagonal restriction
    just sim diag_sheaf

# Neural Diagonal Sheaf orchestrator
nds:
    # Neural Diagonal Sheaf Channel Charting - neural network enhanced diagonal sheaf
    just sim neural_diag_sheaf

# Personalized Federated orchestrator
pfed:
    # Personalized Federated Channel Charting - federated with personalization
    just sim personalized_federated

# Vanilla (baseline) orchestrator
vanilla:
    # Vanilla Channel Charting - baseline siamese network without orchestration
    just sim vanilla

# Run all orchestrators
sim-all: bun cs ds nds fb ot fed pfed vanilla
