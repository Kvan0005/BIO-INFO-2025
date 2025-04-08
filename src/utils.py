# Import necessary libraries
import sys
import random
from anndata import AnnData
import scanpy
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

def density_downsampling(df: pd.DataFrame, od: float, td: float) -> pd.DataFrame:
    """

    cf: https://doi.org/10.1038/s41587-019-0071-9
    Args:
        df (pd.DataFrame): _description_
        od (float): _description_
        td (float): _description_

    Returns:
        pd.DataFrame: _description_
    """
    print(df)
    #? Computes condensed pairwise Euclidean distances between all rows (samples)
    #? Converts condensed vector into a square distance matrix of shape (n, n)
    #? dist_m[i][j] contains the Euclidean distance between point i and j
    dist = distance.pdist(df, metric='euclidean')
    dist_matrix = distance.squareform(dist)

    #? sort the the matrix based on the axis=1 (row-wise) thus each line (represents the distance between i and his closest neighbors)
    #? we take the median of the second closest neighbor (index 1) because the first one is the point itself (distance 0)
    sorted_dist_m = np.sort(dist_matrix , axis=1)
    median_min_dist = np.median(sorted_dist_m[:,1])
    dist_threshold = np.max(median_min_dist)
    
    local_density = np.sum(dist_matrix < dist_threshold, axis=0) #! we removed the "1*" for converting to int
    
    
    return df

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the input data to a range between 0 and 1.
    
    Parameters
    ----------
    df:  
        type: pandas.DataFrame
        summary: The input DataFrame to be normalized.
    
    Returns:
        type: pandas.DataFrame
        summary: The normalized DataFrame (values between 0 and 1).
    """
    return (df - df.min()) / (df.max() - df.min())

# Preprocessing code
def preprocessing(df: pd.DataFrame, od: float, td: float) -> pd.DataFrame:
    """
    Preprocess the DataFrame by filtering out rows with NaN values and applying a threshold.
    
    Parameters
    ----------
    df : 
        name: data
        type: pd.DataFrame
        summary: The input DataFrame to be preprocessed.
    od : 
        name: Outlier Density Threshold
        type: float
        summary: OD represents the quantile threshold for lower-density outliers in the dataframe.
    td : 
        name: Target Density Threshold
        type: float
        summary: OD represents the quantile threshold for higher-density outliers in the dataframe.

    Returns:
    DataFrame filtered by the thresholds
    -------
    """
    
    df = density_downsampling(df, od, td)
    #df = normalize(df)
    return df


if __name__ == "__main__":
    # test the functions
    df = pd.read_csv("/home/linsfa/Documents/BIO-INFO-2025/src/data/fig6_fig7/bone-marrow-mesenchyme-erythrocyte-differentiation_mca.rds.csv", index_col=0)
    df = preprocessing(df, 0.1, 0.9)