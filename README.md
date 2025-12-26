# Commonality inferences

Code and data for reproducing the analyses in the paper.

## Requirements

**Python**: `pip install -r requirements.txt`

**R**: `install.packages(c('lme4', 'lmerTest', 'reticulate'))`

**Quarto**: https://quarto.org/

## Reproducing Analyses

### Behavioral analyses (Figures 2, 3, 5)

```bash
quarto render analysis/behavioral_analyses.qmd
```

### Computational model (Figure 4)

```bash
quarto render analysis/model_analyses.qmd
```

## Structure

- `analysis/` - Quarto notebooks reproducing all analyses
  - `behavioral_analyses.qmd` - Mixed-effects models and Figures 2, 3, 5
  - `model_analyses.qmd` - Bayesian factor model and Figure 4
- `data/` - Experimental data
- `models/` - Computational model code
