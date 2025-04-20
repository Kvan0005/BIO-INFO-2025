import sys
import random
import anndata
import scanpy_modified as scanpy
from ripser import Rips
import umap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.stats import rankdata, entropy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import pairwise_distances, pairwise
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sknetwork.clustering import Louvain
from scipy.spatial import distance
from math import floor
FEATURES_MAX_THRESHOLD = 5 #? 5 is given in the paper dc about logic

def metric_distribution_of_pairwise_distances(df, num_bins: int, visualize: bool = False) -> np.number | np.ndarray:
    """
    ref: https://doi.org/10.1371/journal.pcbi.1011866
    pg: 5
    title: Scoring metric 1 -distribution of pairwise distances
    """
    size_df = len(df)
    df = anndata.AnnData(df)
    df.uns['iroot'] = 0
    
    scanpy.pp.neighbors(df,n_neighbors=max(10,int(0.005*size_df)), method='umap',knn=True)
    scanpy.tl.diffmap(df)
    scanpy.tl.dpt(df)
    tmp = np.stack(df.obs['dpt_distances'])  # type: ignore dont know how to fix this warning
    tmp[tmp == np.inf] = 1.5 * np.max(tmp[tmp != np.inf]) 
    tmp[tmp == -np.inf] = -1 * np.min(tmp[tmp != -np.inf]) 
    a = plt.hist(tmp[np.triu(tmp, 1) != 0], bins = num_bins)
    hs = a[0]/np.sum(a[0])
   
    ent = entropy(hs, base=num_bins)
    if visualize:
        plt.show()
    else:
        plt.close()
    return  ent

def metric_persistent_homology(df, num_bins: int, visualize: bool = False) -> np.number | np.ndarray:
    """
    ref: https://doi.org/10.1371/journal.pcbi.1011866
    pg: 5
    title: Scoring metric 2 -persistent homology
    """  
    df = anndata.AnnData(df)
    df.uns['iroot'] = 0
    scanpy.pp.neighbors(df)
    scanpy.tl.diffmap(df)
    scanpy.tl.dpt(df)
    tmp = np.stack(df.obs['dpt_distances'])  # type: ignore dont know how to fix this warning
    # get line with inf values
    tmp[tmp == np.inf] = np.random.normal(1.5, 0.1) * np.max(tmp[tmp != np.inf])
    tmp[tmp == -np.inf] = -1 * np.min(tmp[tmp != -np.inf])
    rips = Rips(maxdim=0, verbose=False)  # Initialize a Rips object to perform topological analysis of distances. maxdim=0 means we are only interested in connected components in the data (i.e., groups of linked cells).
    crs_tmp = sparse.csr_matrix(tmp) # mapping the data matrix to a (position, distance) matrix
    diagrams= rips.fit_transform(crs_tmp, distance_matrix=True)
    if diagrams is None:
        exit() # for code sanity
    histogram = plt.hist(diagrams[0][:-1,1], bins=num_bins)
    normalize_histogram = histogram[0]/np.sum(histogram[0])
    entropy_value = entropy(normalize_histogram, base=num_bins)
    entropy_value = np.log(entropy_value)
    plt.show() if visualize else plt.close()
    return entropy_value

def metric_vector(df, cells_per_cluster:int, metric: str) -> float:
    df = PCA(n_components=FEATURES_MAX_THRESHOLD).fit_transform(df)
    _number_of_clusters = min(floor(len(df)/cells_per_cluster), 100)
    
    
if __name__ == "__main__":
    # Example usage
    from utils import preprocessing
    # path = "/home/linsfa/Documents/BIO-INFO-2025/src/data/fig6_fig7/"
    # for file in os.listdir(path):
    #     if file.endswith(".rds.csv"):
    #         print(file)
    #         df = pd.read_csv(path + file, index_col=0)
    #         df = preprocessing(df, 0.1, 0.9)
    #         metric_persistent_homology(df, num_bins=10)
    path = "/home/linsfa/Documents/BIO-INFO-2025/src/data/fig6_fig7/NKT-differentiation_engel.rds.csv"
    df = pd.read_csv(path, index_col=0)
    df = preprocessing(df, 0.1, 0.9)
    print(metric_persistent_homology(df, num_bins=10))
    