"""
Common imports and setup for Quarto notebooks using knitr engine.

Each Python chunk should start with:
    exec(open('_setup.py').read())
or:
    from _setup import *
"""

import sys
from pathlib import Path

# Find project root
base_dir = Path.cwd().resolve()
while base_dir.name and not (base_dir / 'data').exists():
    base_dir = base_dir.parent

data_dir = base_dir / 'data'
figures_dir = base_dir / 'outputs' / 'figures'
figures_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(base_dir))

# Common imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, linregress

# Color palette
COLORS = {
    'opposing': '#648FFF',
    'shared': '#DC267F',
    'human': '#2c3e50',
    'bayesian': '#648FFF',
    'egocentric': '#e07a5f',
    'scrambled': '#95a5a6',
    'focal': '#2E86AB',
    'same': '#A23B72',
    'diff': '#95a5a6',
}
lowhigh_palette = ['#648FFF', '#DC267F']
