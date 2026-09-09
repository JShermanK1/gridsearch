# %%
import glob
import os
import pickle
import time
from typing import Literal
# os.environ["SCIPY_ARRAY_API"] = "1"

import anndata as ann
import colorcet as cc
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import sklearn.base as skbase
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.pipeline as pipe
from matplotlib import pyplot as plt
from sklearn.experimental import enable_halving_search_cv
import scipy.spatial as sps
from cuml.metrics.cluster import silhouette_score
from cuml.decomposition import PCA
from cuml.neighbors import NearestNeighbors
import cugraph as cg
import cupy as cp
from cudf import DataFrame



class ScNeighbors(NearestNeighbors, skbase.TransformerMixin, skbase.BaseEstimator):
        
    def transform(self, X):
        neighbor_list: DataFrame
        distance_list: DataFrame
        distance_list, neighbor_list = self.kneighbors()
        neighbor_list = neighbor_list.reset_index(
            drop= True,
        ).reset_index(
        ).melt(
            id_vars= "index",
            value_name= "destination",
        ).join(
            distance_list.reset_index(
                drop= True,
            ).melt(
                value_name= "dist"
            )["dist"]
        )
        neighbor_list = neighbor_list[["index", "destination", "dist"]]
        map = X.index.to_series().reset_index(drop= True)
        neighbor_list["index"] = neighbor_list["index"].map(map)
        neighbor_list["destination"] = neighbor_list["destination"].map(map)
        return (neighbor_list, X)

    
class ScLeiden(skbase.ClassifierMixin, skbase.BaseEstimator):
    def __init__(self, resolution= 1):
        self.resolution = resolution


    def fit(self, X, y= None):
        g = cg.from_cudf_edgelist(X[0], source= "index", edge_attr= "dist")
        parts, _ = cg.leiden(
            g,
            resolution= self.resolution,
        )
        self.classes_ = parts.set_index("vertex").join(X[1])
        return self
    
    def predict(self, X, y= None):
        return self.classes_

    def score(self, X, y= None):
        classes = self.classes_["partition"]
        X = self.classes_.drop(columns= "partition")
        return silhouette_score(
            X,
            labels= classes,
        ) * cp.log10(X.shape[1])

        

if __name__ == "__main__":
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
    import argparse
    import pathlib
    from tempfile import TemporaryDirectory
    import joblib
    from typing import Tuple
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
    import zarr
    import cudf as cdf
    import dask
    import dask.dataframe as dd
    import dask_ml.model_selection as dms
    import sys
    import sklearn
    import itertools

    dask.config.set({
        "array.backend": "cupy",
        "dataframe.backend": "cudf",
        })
    # sklearn.set_config(array_api_dispatch= True)


    rng = np.random.default_rng(0)


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
    def range_input(string: str) -> Tuple[float, float]:
        """
        converts string of comma seperated floats into a tuple to set gridsearch params
        """
        range_tuple = tuple([float(val) for val in string.replace(" ", "").split(",")])
        assert len(range_tuple) == 2
        return range_tuple

    apr.add_argument(
        "--res_limits", "-r",
        type= float,
        nargs= 2,
        default= [0.5, 1.5],
    )
    apr.add_argument(
        "--nn_limits", "-n",
        type= int,
        nargs= 2,
        default= [20, 40],
    )
    apr.add_argument(
        "--comp_limits", "-c",
        type= int,
        nargs= 2,
        default= [15, 35],
    )
    apr.add_argument(
        "--mem", "-m",
        type= str,
        default= "12G",
        help= "Size of RMM pool allocation for each gpu",
    )
    apr.add_argument(
        "--transpose", "-t",
        action= "store_true",
    )

    args = apr.parse_args()
    if args.prefix:
        args.prefix += "-"


    cluster = LocalCUDACluster(
        protocol= "ucx",
        enable_infiniband= True,
        # enable_tcp_over_ucx= True,
        # enable_rdmacm= True,
        rmm_pool_size= args.mem,
        rmm_allocator_external_lib_list= ["cupy"],
        # rmm_async= True,
    )
    client = Client(
        cluster,
    )   

    if args.transpose:
        merged_data = cdf.read_parquet(
                args.anndata,
            ).astype("float32")
        merged_data = merged_data.T.copy(deep= True)
    else:
        merged_data = cdf.read_parquet(
                args.anndata,
            ).astype("float32")
    

    print("fitting gridsearch")

    n_components = np.linspace(*args.comp_limits, 21, dtype= int)
    n_neighbors = itertools.cycle([np.linspace(*args.nn_limits, 21, dtype= int)])
    resolution = itertools.cycle([np.linspace(*args.res_limits, 21)])

    grid = list(zip(n_components, n_neighbors, resolution))
    # %%
    def score(params, data):
        nn = params[1]
        r = params[2]
        pca = PCA(
            n_components= params[0]
        )
        X_pca = pca.fit_transform(data)
        scores = cp.empty((nn.shape[0], r.shape[0]))
        for i, n_neigh in enumerate(nn):
            scneighbors = ScNeighbors(
                n_neighbors= n_neigh
            )
            X_neighbor = scneighbors.fit_transform(X_pca)
            for j, res in enumerate(r):
                scleid = ScLeiden(
                    resolution= res
                )
                scleid.fit(X_neighbor)
                scores[i, j] = scleid.score(X_neighbor)
        return scores

    with joblib.parallel_backend("dask"):
        result = joblib.Parallel(verbose= 100)(
            joblib.delayed(score)(params, merged_data.copy(deep= False)) for params in grid
        )

    result = np.array([
        arr.get() for arr in result
    ])

    idx = pd.MultiIndex.from_product(
        [n_components, n_neighbors, resolution],
        names= ["n_components", "n_neighbors", "resolution"]
    )
    grids_df = pd.Series(
        result.flat,
        index= idx,
        name= "score"
    ).to_frame(
    ).reset_index(
    ).to_parquet(f"pickles/{args.prefix}dask")
    # %%
