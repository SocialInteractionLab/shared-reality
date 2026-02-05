# Commonality inferences

Code and data for reproducing the analyses in the paper.

## Requirements

- **Python** 3.10+ with scientific packages (numpy, pandas, scipy, matplotlib, seaborn, jax)
- **R** 4.0+ with packages: `lme4`, `lmerTest`, `dplyr`, `tidyr`, `reticulate`
- **Quarto** 1.3+: <https://quarto.org/>

## Setup

### Option 1: Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles virtual environments automatically.

```bash
# Install Python dependencies (creates .venv automatically)
uv sync

# Install R packages
Rscript -e "install.packages(c('lme4', 'lmerTest', 'dplyr', 'tidyr', 'purrr', 'reticulate'))"
```

### Option 2: Using pip

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install R packages
Rscript -e "install.packages(c('lme4', 'lmerTest', 'dplyr', 'tidyr', 'purrr', 'reticulate'))"
```

## Reproducing Analyses

The Quarto notebooks use R's `reticulate` package to run Python code. You must tell R which Python to use by setting the `RETICULATE_PYTHON` environment variable.

### Quick start

```bash
# Set Python path and render all notebooks
RETICULATE_PYTHON=$(pwd)/.venv/bin/python quarto render analysis/model_analyses.qmd
RETICULATE_PYTHON=$(pwd)/.venv/bin/python quarto render analysis/behavioral_analyses.qmd
RETICULATE_PYTHON=$(pwd)/.venv/bin/python quarto render analysis/supplement.qmd
```

### Alternative: Export once per session

```bash
# Export the variable for your shell session
export RETICULATE_PYTHON=$(pwd)/.venv/bin/python

# Then render notebooks without prefix
quarto render analysis/model_analyses.qmd
quarto render analysis/behavioral_analyses.qmd
quarto render analysis/supplement.qmd
```

### Individual notebooks

| Notebook | Description | Main outputs |
|----------|-------------|--------------|
| `behavioral_analyses.qmd` | Mixed-effects models | Figure 2, Figure 3, Figure 4 |
| `model_analyses.qmd` | Bayesian factor model, LLM analysis | Figure 5, Figure 6 |
| `supplement.qmd` | Supplementary analyses | SI figures and tables |

## Troubleshooting

**"Python specified in RETICULATE_PYTHON does not exist"**
- Make sure you ran `uv sync` or `pip install -r requirements.txt` first
- Check the path: `ls -la .venv/bin/python`
- Use absolute path: `export RETICULATE_PYTHON=/full/path/to/shared-reality/.venv/bin/python`

**R can't find packages**
- Install missing R packages: `Rscript -e "install.packages('package_name')"`

**JAX errors on Apple Silicon**

- JAX should work out of the box on M1/M2 Macs with the dependencies specified

## Structure

```text
├── analysis/           # Quarto notebooks
│   ├── behavioral_analyses.qmd # Mixed-effects models (Figures 2, 3, 4)
│   ├── model_analyses.qmd      # Bayesian factor model, LLM (Figures 5, 6)
│   └── supplement.qmd          # Supplementary analyses (SI figures/tables)
├── data/               # Experimental data
│   ├── responses.csv   # Main behavioral data
│   ├── questions.csv   # Survey questions and norms
│   ├── messages.csv    # Conversation transcripts
│   └── llm_results/    # LLM prediction outputs
├── models/             # Computational model code
│   ├── model.py        # Bayesian factor model
│   └── llm/            # LLM prediction pipeline
├── outputs/figures/    # Generated figures
├── pyproject.toml      # Python dependencies (for uv)
└── requirements.txt    # Python dependencies (for pip)
```
