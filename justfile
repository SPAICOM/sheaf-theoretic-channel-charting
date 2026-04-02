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

# Bundle orchestrator
bun *args:
    # Bundle Channel Charting - bundle-based approach
    just sim bundle {{args}}

# Cover Sheaf orchestrator
cs *args:
    # Cover Sheaf Channel Charting - sheaf-based with cover decomposition
    just sim cover_sheaf {{args}}

# Diagonal Sheaf orchestrator
ds *args:
    # Diagonal Sheaf Channel Charting - sheaf with diagonal restriction
    just sim diag_sheaf {{args}}

# Federated learning orchestrator
fed *args:
    # Federated Channel Charting - standard federated learning approach
    just sim federated {{args}}

# Flat Bundle orchestrator
fb *args:
    # Flat Bundle Channel Charting - bundle-based approach without sheaf structure
    just sim flat_bundle {{args}}

# Neural Diagonal Sheaf orchestrator
nds *args:
    # Neural Diagonal Sheaf Channel Charting - neural network enhanced diagonal sheaf
    just sim neural_diag_sheaf {{args}}

# Optimal Transport orchestrator
ot *args:
    # Optimal Transport Channel Charting - uses optimal transport for aggregation
    just sim optimal_transport {{args}}

# Personalized Federated orchestrator
pfed *args:
    # Personalized Federated Channel Charting - federated with personalization
    just sim personalized_federated {{args}}

# Vanilla (baseline) orchestrator
vanilla *args:
    # Vanilla Channel Charting - baseline siamese network without orchestration
    just sim vanilla {{args}}

# Single-agent training (run without orchestrator)
single *args:
    # Single-agent Channel Charting - runs single agent directly
    uv run scripts/single_agent_simulation.py {{args}}

# Evaluate all orchestrators and save metrics to results/eval_metrics.parquet
eval *args:
    uv run scripts/eval.py {{args}}

# Run all orchestrators (alphabetically ordered)
sim-all *args:
    just sim bundle {{args}}
    just sim cover_sheaf {{args}}
    just sim diag_sheaf {{args}}
    just sim federated {{args}}
    just sim flat_bundle {{args}}
    just sim neural_diag_sheaf {{args}}
    just sim optimal_transport {{args}}
    just sim personalized_federated {{args}}
    just sim vanilla {{args}}
