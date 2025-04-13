import sys
import random
from anndata import AnnData
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



def metrics_distribution_of_pairwise_distances(df, num_bins: int, visualize: bool = True) -> np.number | np.ndarray:
    """
    ref: https://doi.org/10.1371/journal.pcbi.1011866
    pg: 5
    title: Scoring metric 1 -distribution of pairwise distances
    """
    size_df = len(df)
    df = AnnData(df)
    df.uns['iroot'] = 0
    
    scanpy.pp.neighbors(df,n_neighbors=max(10,int(0.005*size_df)), method='umap',knn=True)
    scanpy.tl.diffmap(df)
    scanpy.tl.dpt(df)
    tmp = np.stack(df.obs['dpt_distances'])
    tmp[tmp == np.inf] = 1.5 * np.max(tmp[tmp != np.inf]) 
    tmp[tmp == -np.inf] = -1 * np.min(tmp[tmp != -np.inf]) 
    a = plt.hist(tmp[np.triu(tmp, 1) != 0], bins = num_bins)
    hs = a[0]/np.sum(a[0])
   
    ent = entropy(hs, base=num_bins)
    
    if visualize == True:
        plt.show()
    elif visualize == False:
        plt.close()
    return  ent

if __name__ == "__main__":
    # Example usage
    from utils import preprocessing
    df = pd.read_csv("/home/jmjn/bioinformatics/project/BIO-INFO-2025/src/data/fig6_fig7/bone-marrow-mesenchyme-erythrocyte-differentiation_mca.rds.csv", index_col=0)
    df = preprocessing(df, 0.1, 0.9)
    metrics_distribution_of_pairwise_distances(df, num_bins=10)
    
    