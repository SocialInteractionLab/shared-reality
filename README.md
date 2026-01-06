# Commonality inferences

Code and data for reproducing the analyses in the paper.

## Requirements

- **Python** 3.10+ with scientific packages (numpy, pandas, scipy, matplotlib, seaborn, jax)
- **R** with packages: `lme4`, `lmerTest`, `reticulate`
- **Quarto**: <https://quarto.org/>

## Setup

### Option 1: Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles dependencies automatically.

```bash
# Install Python dependencies
uv sync

# Install R packages
Rscript -e "install.packages(c('lme4', 'lmerTest', 'reticulate'))"
```

### Option 2: Using pip (pure Python)

If you prefer not to use uv, you can install dependencies with pip:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install numpy pandas scipy matplotlib seaborn statsmodels polars pyarrow jax jaxlib tabulate

# Install R packages
Rscript -e "install.packages(c('lme4', 'lmerTest', 'reticulate'))"
```

## Reproducing Analyses

### Computational model (Figure 4)

```bash
quarto render analysis/model_analyses.qmd
```

### Behavioral analyses (Figures 2, 3)

```bash
quarto render analysis/behavioral_analyses.qmd
```

### Supplement

```bash
quarto render analysis/supplement.qmd
```

## Structure

``` text
├── analysis/           # Quarto notebooks
│   ├── model_analyses.qmd      # Bayesian factor model (Figure 4)
│   ├── behavioral_analyses.qmd # Mixed-effects models (Figures 2, 3)
│   └── supplement.qmd          # Supplementary analyses
├── data/               # Experimental data
│   ├── responses.csv   # Main behavioral data
│   ├── questions.csv   # Survey questions
│   └── llm_results/    # LLM prediction outputs
├── models/             # Computational model code
│   ├── model.py        # Bayesian factor model
│   └── llm/            # LLM prediction pipeline
└── pyproject.toml      # Python dependencies
```
