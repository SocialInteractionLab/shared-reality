# Demos

Interactive visualizations for the shared reality computational models.

## Bayesian Factor Model Demo

`bayesian_update_demo.py` is an interactive [marimo](https://marimo.io/) notebook that visualizes how the Bayesian factor model updates beliefs about a conversation partner after observing a single response.

### Running the notebook

1. **Install dependencies** (if you haven't already):
   ```bash
   uv sync
   ```

2. **Run the notebook**:
   ```bash
   uv run marimo run demos/bayesian_update_demo.py
   ```

   This will open the notebook in your browser at `http://localhost:2718`.

3. **Edit mode** (optional): To modify the notebook interactively:
   ```bash
   uv run marimo edit demos/bayesian_update_demo.py
   ```

### What the demo shows

- **Prior, Likelihood, Posterior**: Visualizes Bayesian updating in latent factor space (θ-space)
- **Pairwise prediction**: Shows P(response=1,2,3,4,5) for any target question given an observation
- **Parameter exploration**: Adjust k (number of factors), σ_prior, observed question, and response

### Requirements

- Python 3.10+
- Dependencies listed in `pyproject.toml` (installed via `uv sync`)
