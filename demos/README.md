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

- **Model equations**: Formal specification of both Bayesian and Egocentric models
- **Prior, Likelihood, Posterior**: Visualizes Bayesian updating in latent factor space (θ-space)
- **Side-by-side model comparison**: Compare predictions from Bayesian (population structure) vs Egocentric (self-structure) models
- **Pairwise prediction**: Shows P(response=1,2,3,4,5) for any target question from both models + mixture
- **Parameter exploration**: Adjust k, σ_prior, λ (mixture weight), observed question, response, and belief profile

### Requirements

- Python 3.10+
- Dependencies listed in `pyproject.toml` (installed via `uv sync`)
