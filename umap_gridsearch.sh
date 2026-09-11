#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate gridsearch
python scripts/umap_gridsearch.py \
"$@"
