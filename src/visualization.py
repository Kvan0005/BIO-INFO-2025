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
from tqdm import tqdm

from utils import preprocessing


def sc_vis(file):
    print('reding ... {}'.format(file))
    df = pd.read_csv(file, index_col=0)
    df = preprocessing(df, 0.05, 1)
    print('size of the file is {}'.format(df.shape))
    data = np.array(df)
    data = anndata.AnnData(data)
    data.uns['iroot'] = 0 
    plt.figure(figsize=(8,8))
    scanpy.set_figure_params(dpi=80, dpi_save=150, figsize=(5,5)) #TODO checkpoint
    scanpy.tl.pca(data, svd_solver='arpack')
    scanpy.pp.neighbors(data)
    scanpy.tl.diffmap(data)
    scanpy.tl.dpt(data)
    scanpy.tl.diffmap(data,color=['dpt_pseudotime'])
    
    pca = PCA(n_components=20)
    embedding_pca = pca.fit_transform(df)
    A = kneighbors_graph(embedding_pca, 10, mode='connectivity', include_self=True)
    louvain = Louvain()
    labels = louvain.fit_transform(A)
    Umap = umap.UMAP()
    embedding_umap = Umap.fit_transform(embedding_pca)
    fig = plt.figure(figsize=(5,5))
    plt.scatter(embedding_umap[:,0],embedding_umap[:,1],c=labels, alpha=1, cmap='tab20')
    plt.xticks([])
    plt.yticks([])
    plt.grid(b=None)
    plt.show()
    #plt.savefig('{}.png'.format(file))
    
if __name__ == '__main__':
    file = "data/fig6_fig7/bone-marrow-mesenchyme-erythrocyte-differentiation_mca.rds.csv"
    sc_vis(file)