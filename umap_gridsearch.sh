#! /bin/bash

eval "$(conda shell.bash hook)"
conda activate gridsearch
python umap_gridsearch.py \
"$@"
