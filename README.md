# Commonality inferences

Code and data for reproducing the analyses in the paper.

## Requirements

- **uv** (Python package manager): <https://docs.astral.sh/uv/>
- **R** with packages: `lme4`, `lmerTest`, `reticulate`
- **Quarto**: <https://quarto.org/>

## Setup

```bash
# Install Python dependencies
uv sync

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
