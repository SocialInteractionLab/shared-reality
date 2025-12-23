# Shared Reality Through Cooperative Communication

Code and data for reproducing the analyses in the paper.

## Requirements

**Python**: `pip install -r requirements.txt`

**R**: `install.packages(c('lme4', 'lmerTest', 'reticulate'))`

**Quarto**: https://quarto.org/

## Reproducing Analyses

### Behavioral analyses (Figures 2, 3, 5)

```bash
cd analysis
quarto render paper_analyses.qmd
```

### Computational model (Figure 4)

```bash
python scripts/run_model_comparison.py
python scripts/generate_figures.py
```

## Structure

- `analysis/` - Quarto document reproducing all behavioral statistics
- `data/` - Experimental data
- `models/` - Computational models
- `scripts/` - Analysis scripts
- `outputs/` - Generated figures and model outputs
