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
from metrics import metric_distribution_of_pairwise_distances, metric_persistent_homology, metric_vector, metric_ripley_dpt, metric_avg_connection 
from utils2 import preprocessing

def original_scoring(df, num_downsample= 5000):
    if len(df) > num_downsample:
        tmp = []
        for i in range(3):
            np.random.seed(i)
            random.seed(i)
            df = df[random.sample(range(len(df)), num_downsample),:]
            tmp.append(calculate_metrics(df)) 
        scores = list(np.median(np.stack(tmp),axis=0))
    else:
        scores = calculate_metrics(df)
    return scores

def our_scoring(df: np.ndarray, num_downsample= 5000):
    if len(df) > num_downsample:
        tmp = []
        for i in range(3):
            np.random.seed(i) # for reproducibility but not necessary due to seed being set in the function
            sampled_df = df[np.random.choice(df.shape[0], num_downsample, replace=True),:] 
            tmp.append(calculate_metrics(sampled_df))
        scores = list(np.median(np.stack(tmp),axis=0))
    else:
        scores = calculate_metrics(df)
    return scores

def calculate_metrics(df: np.ndarray) -> list[np.floating | float | np.number]:
    df = preprocessing(df, 0.05, 1)
    # sc1 = metric_distribution_of_pairwise_distances(df, num_bins = 10)
    # sc2 = metric_persistent_homology(df,num_bins = 3)
    sc3 = metric_vector(df)
    # sc4 = metric_ripley_dpt(df)
    # sc5 = metric_avg_connection(df)
    return [0,0,sc3,0,0]
    # return [sc1,sc2,sc3,sc4,sc5]
    
    
def explain_score(sc_traj_clstr_score):
    META_SCORES = list(np.load('data/simulated_metascores_12000.npy')) # Loading pre-computed scores for simulated datasets

    clstr = META_SCORES[:3000]
    traj = META_SCORES[3000:6000]
    clstr_r1 = META_SCORES[6000:9000]
    traj_r1 = META_SCORES[9000:]
   

    npy_sim = np.array(META_SCORES)
    feature_names = ['P-dist','Homology','Vector','Ripleys','Deg. of Sep.']

    metric = 'euclidean'
    seed = 1
    n_neighbors = 50
    min_dist = 0.6
    figsize = 5

    scaler = StandardScaler()
    tmp_np = scaler.fit_transform(npy_sim)
    tmp_reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2,random_state=seed,min_dist=min_dist, metric=metric)
    embedding = tmp_reducer.fit_transform(tmp_np)
    c = [0]*3000 + [1]*3000 + [0]*3000 + [1]*3000

    neigh = KNeighborsClassifier(n_neighbors=100)
    neigh.fit(embedding, c) #type: ignore

    p = neigh.predict_proba(embedding)[:,1] #type: ignore

    ####################
    ####################
    INPUT_SCALED = scaler.transform(np.array(sc_traj_clstr_score).reshape(1, -1))
    UMAP_PROJECTION = tmp_reducer.transform(INPUT_SCALED.reshape(1, -1))    
    ####################
    ####################

    plt.figure(figsize=(figsize,figsize))
    plt.scatter(embedding[:,0],embedding[:,1], c = p, alpha = 1) #type: ignore
    plt.scatter(UMAP_PROJECTION[0][0],UMAP_PROJECTION[0][1],color='red',marker='^') #type: ignore
    plt.show()

    for i in range(len(feature_names)):
        feat = i
        plt.figure(figsize=(5,3))
        plt.violinplot([np.array(clstr)[:,feat],
                        np.array(clstr_r1)[:,feat],
                        np.array(traj_r1)[:,feat],
                        np.array(traj)[:,feat]],
                      showmeans = True, showextrema=False)
        plt.axhline(sc_traj_clstr_score[i],c='red',ls='--')
        plt.title(feature_names[i], fontsize = 20)
        plt.xticks(fontsize=15, rotation=315)
        plt.xticks([1, 2, 3, 4], ['Clear Clusters','Noisy Clusters','Noise Trajectory','Clear Trajectory'])
        plt.show()
    

if __name__ == '__main__':
    file = "data/fig6_fig7/bone-marrow-mesenchyme-erythrocyte-differentiation_mca.rds.csv"
   