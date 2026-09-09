#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate rapids_singlecell
python scripts/umap_gridsearch.py \
"$@"
