# %%
import os
from typing import Literal

import numpy as np
import pandas as pd
from cuml.metrics.cluster import silhouette_score
from cuml.decomposition import (PCA, IncrementalPCA)
from cuml.neighbors import NearestNeighbors
import cugraph as cg
import cupy as cp
from cudf import DataFrame
from sklearn import base as skbase



class ScNeighbors(NearestNeighbors, skbase.TransformerMixin, skbase.BaseEstimator):
        
    def transform(self, X):
        neighbor_list: DataFrame
        distance_list: DataFrame
        distance_list, neighbor_list = self.kneighbors()
        neighbor_list = neighbor_list.reset_index(
        ).melt(
            id_vars= "index",
            value_name= "destination",
        ).join(
            distance_list.melt(
                value_name= "dist"
            )["dist"]
        )
        neighbor_list = neighbor_list[["index", "destination", "dist"]]
        return neighbor_list

    
class ScLeiden(skbase.ClassifierMixin, skbase.BaseEstimator):
    def __init__(self, resolution= 1):
        self.resolution = resolution


    def fit(self, X, y= None):
        g = cg.from_cudf_edgelist(X, source= "index", edge_attr= "dist")
        classes, _ = cg.leiden(
            g,
            resolution= self.resolution,
        )
        self.classes_ = classes.sort_values("vertex")["partition"]
        return self
    
    def predict(self, X, y= None):
        return self.classes_

    def score(self, X, y= None):
        return silhouette_score(
            X,
            labels= self.classes_,
        ) * cp.log(X.shape[1])

        

if __name__ == "__main__":
    import argparse
    import pathlib
    import joblib
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
    import cudf as cdf
    import dask
    import itertools

    dask.config.set({
        "array.backend": "cupy",
        "dataframe.backend": "cudf",
        })


    rng = np.random.default_rng(0)

# %%


    apr = argparse.ArgumentParser(
        description= """
        Gridsearch for clustering arugments that optimize adjusted silhouette score.
        VRAM requirements scale very quickly with number of features and number of observations.
        It is recommended to filter down to a number of highly variable features that will fit in memory for your device.
        Approximately 100,000 obs x 5,000 features will fit in 40G of VRAM.
        """,
    )
    apr.add_argument(
        "--input", "-i",
        type= pathlib.Path,
        required= True,
        help= """
        Required:     
        A parquet of observations x variables.
        """
    )
    apr.add_argument(
        "--output", "-o",
        type= pathlib.Path,
        required= True,
        help= """
        Required:     
        File path for output parquet. A long form parquet with columns for the number of PCA components, number of nearest neighbors, resolution and resulting score.
        """,
    )
    apr.add_argument(
        "--res_limits", "-r",
        type= float,
        nargs= 2,
        default= [0.5, 1.5],
        help= """
        Default 0.5 1.5:    
        Two space seperated values for the lower and upper limits for resolutions. 21 evenly spaced values (inclusive) are tested
        """,
    )
    apr.add_argument(
        "--nn_limits", "-n",
        type= int,
        nargs= 2,
        default= [20, 40],
        help= """
        Default 20 40:    
        Two space seperated values for the lower and upper limits for nearest neighbors to test.
        """,
    )
    apr.add_argument(
        "--comp_limits", "-c",
        type= int,
        nargs= 2,
        default= [15, 35],
        help= """
        Default 15 35:    
        Two space seperated values for the lower and upper limits for number of PCA components to test.
        """,
    )
    apr.add_argument(
        "--mem", "-m",
        type= str,
        default= "12G",
        help= """
        Default 12G:    
        Initial size of RMM pool allocation for each gpu. The pool may increase past this if necessary.
        """,
    )
    apr.add_argument(
        "--transpose", "-t",
        action= "store_true",
        help= "Transpose the parquet before testing",
    )

    args = apr.parse_args()
# %%
    args.input.resolve(strict= True)

    os.makedirs(
        args.output.resolve().parent,
        exist_ok= True,
    )


    print("fitting gridsearch")

    n_components = np.linspace(
        *args.comp_limits, 
        np.abs(args.comp_limits[1] - args.comp_limits[0]) + 1, 
        dtype= int
    )
    n_neighbors = np.linspace(
        *args.nn_limits, 
        np.abs(args.nn_limits[1] - args.nn_limits[0]) + 1, 
        dtype= int
    )
    resolution = np.linspace(*args.res_limits, 21)

# %%
    grid = zip(
        n_components, 
        itertools.cycle([n_neighbors]), 
        itertools.cycle([resolution])
    )
    # %%
    def score(params, file):
        if args.transpose:
            data = cdf.read_parquet(
                    file,
                ).astype("float32").T.values
        else:
            data = cdf.read_parquet(
                    file,
                ).astype("float32").values
        nn = params[1]
        r = params[2]
        pca = IncrementalPCA(
            n_components= params[0]
        )
        X_pca = pca.fit_transform(data)
        scores = cp.empty((nn.shape[0], r.shape[0]))
        for i, n_neigh in enumerate(nn):
            scneighbors = ScNeighbors(
                n_neighbors= n_neigh,
                output_type= "cudf",
            )
            X_neighbor = scneighbors.fit_transform(X_pca)
            for j, res in enumerate(r):
                scleid = ScLeiden(
                    resolution= res
                )
                scleid.fit(X_neighbor)
                scores[i, j] = scleid.score(X_pca)
        return scores

    with LocalCUDACluster(
        protocol= "ucx",
        enable_infiniband= True,
        rmm_pool_size= args.mem,
        rmm_allocator_external_lib_list= ["cupy"],
    ) as cluster, Client(cluster)  as client: 

        with joblib.parallel_backend("dask"):
            result = joblib.Parallel(verbose= 100)(
                joblib.delayed(score)(params, args.input) for params in grid
            )

        result = np.array([
            arr.get() for arr in result
        ])

    print("got result")
    idx = pd.MultiIndex.from_product(
        [n_components, n_neighbors, resolution],
        names= ["n_components", "n_neighbors", "resolution"]
    )
    print("made index")
    grids_df = pd.Series(
        result.flat,
        index= idx,
        name= "score"
    ).to_frame(
    ).reset_index(
    ).to_parquet(args.output)
    print("made parquet")
    # %%
