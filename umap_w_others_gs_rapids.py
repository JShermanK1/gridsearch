# %%
import glob
import os
import pickle
import time

import anndata as ann
import colorcet as cc
import joblib
import numpy as np
import scanpy as sc
import seaborn as sns
import sklearn.base as skbase
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.pipeline as pipe
from matplotlib import pyplot as plt
from sklearn.experimental import enable_halving_search_cv
import scipy.spatial as sps
import argparse
import pathlib
from tempfile import TemporaryDirectory

class ScPCA(skbase.TransformerMixin, skbase.BaseEstimator):
    def __init__(self, mask= None):
        self.mask = mask

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def fit(self, X, y= None):
        self._is_fitted = True
        return self

    def transform(self, X):
        X = rsc.pp.pca(
            X,
            copy= True,
        )
        X.X = None
        return X

class ScNeighbors(skbase.TransformerMixin, skbase.BaseEstimator):
    def __init__(self, n_neighbors= 15, n_pcs= 10):
        self.n_neighbors = n_neighbors
        self.n_pcs = n_pcs

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def fit(self, X, y= None):
        self._is_fitted = True
        return self
        
    def transform(self, X):
        rsc.pp.neighbors(
            X,
            n_neighbors= self.n_neighbors,
            n_pcs= self.n_pcs,
        )
        X.uns["n_pcs"] = self.n_pcs
        return X
    
class ScLeiden(skbase.TransformerMixin, skbase.BaseEstimator):
    def __init__(self, resolution= 1):
        self.resolution = resolution

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def fit(self, X, y= None):
        self._is_fitted = True
        return self

    def transform(self, X):
        rsc.tl.leiden(
            X,
            resolution= self.resolution,
        )
        return X

class ScScore(skbase.TransformerMixin, skbase.BaseEstimator):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def fit(self, X, y= None):
        self._is_fitted = True
        return self

    def score(self, X, y= None, sample_weight= None):
        return silhouette_score(
            X.obsm["X_pca"][:, :X.uns["n_pcs"]],
            labels= X.obs["leiden"].cat.codes,
        ) * np.log10(X.obs["leiden"].nunique() + X.uns["n_pcs"])
    

    

if __name__ == "__main__":
    import cupy as cp
    import cupyx as cpx
    from cupyx.scipy.spatial.distance import pdist
    import rapids_singlecell as rsc
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
    from cuml.metrics.cluster import silhouette_score

    rmm.reinitialize(
        pool_allocator= True,
    )
    cp.cuda.set_allocator(
        rmm_cupy_allocator
    )
    rng = np.random.default_rng(0)

    sns.set_style("whitegrid")
    analysis_layer = None #None == "X"
    # %%
    os.makedirs(
        "figures",
        exist_ok= True,
    )
    os.makedirs(
        "pickles",
        exist_ok= True,
    )
    os.makedirs(
        "data",
        exist_ok= True,
    )
    sns.set_style("whitegrid")

    apr = argparse.ArgumentParser(
        description= "Gridsearch for clustering arugments that optimize silhouette score",
    )
    apr.add_argument(
        "--anndata", "-a",
        type= pathlib.Path,
        required= True,
        help= "input annotated dataframe"
    )
    apr.add_argument(
        "--prefix", "-p",
        type= str,
        default= "",
        help= "prefix for outputs",
    )
    apr.add_argument(
        "--iterations", "-i",
        type= int,
        default= 2,
    )
    apr.add_argument(
        "--layer", "-l",
        type= str,
        default= None,
        help= "AnnData layer to cluster",
    )

    args = apr.parse_args()
    if args.prefix:
        args.prefix += "-"

    analysis_layer = args.layer #None == "X"
# %%
    merged_data = sc.read_h5ad(args.anndata)
    merged_data = ann.AnnData(
        X= merged_data.layers[args.layer],
        obs= merged_data.obs[[]],
        var= merged_data.var[[]],
    )
    # %%
    # %%
    rsc.get.anndata_to_GPU(merged_data, convert_all= True)
    #sample_size = merged_data.n_obs
    #distA = cp.ndarray(sample_size * (sample_size - 1) // 2, dtype= np.float32)

    # %%
    with TemporaryDirectory() as tmpd:
    # %%
        pca = ScPCA()
        neighbors = ScNeighbors()
        scleid = ScLeiden()
        scscorer = ScScore()
        workflow = pipe.make_pipeline(
            pca, neighbors, scleid, scscorer,
            memory= tmpd,
        )
        param_grid = {
            "scneighbors__n_pcs": range(5, 25),
            "scneighbors__n_neighbors": range(20, 50),
            "scleiden__resolution": np.linspace(1, 2, 20) 
        }
        X_train, X_test = skms.train_test_split(
            merged_data,
            test_size= 0.2,
            random_state= 0,
        )

        workflow.score(merged_data)

        # %%

        grids = skms.GridSearchCV(
            workflow,
            param_grid= param_grid,
        ) 

        grids.fit(merged_data)

    # %%
    grids.best_params_

    # %%
    with open(f"pickles/{args.prefix}gridsearch_gpu_1000", "wb") as f:
        pickle.dump(grids, f)

    # %%
    fig, axs = plt.subplots(3)
    for ax, param in zip(axs, param_grid.keys()):
        sns.lineplot(x= grids.cv_results_["param_" + param], y= grids.cv_results_["mean_test_score"], ax= ax)
        ax.set_title(param)
    fig.tight_layout()
    fig.savefig(f"figures/{args.prefix}gpu_params_1000.pdf")

    # %%
    sc.pp.pca(
        merged_data,
        layer= analysis_layer,
    )

    sc.pl.pca_variance_ratio(
        merged_data,
        log= True,
        save= f"{args.prefix}gpu_merged.png"
    )

    # %%

    # %%
    sc.pp.neighbors(
        merged_data,
        n_neighbors= grids.best_params_["scneighbors__n_neighbors"],
        n_pcs= grids.best_params_["scneighbors__n_pcs"],
    )
    sc.tl.umap(
        merged_data,
    )
    sc.tl.leiden(
        merged_data,
        resolution= grids.best_params_["scleiden__resolution"]
    )
    print(skm.silhouette_score(
        merged_data.obsm["X_pca"],
        labels= merged_data.obs["leiden"]
    ))

    # %%
    sc.pl.umap(
        merged_data,
        color= [
            "CDH1",
            "leiden", 
            "tissue_location",
            "disease_timing",
        ],
        gene_symbols= "gene_symbol",
        layer= analysis_layer,
        cmap= "inferno",
        palette= cc.glasbey_category10,
        save= f"{args.prefix}gpu_merged.png",
        vmax= 4,
        ncols= 1,
    )

    # %%
