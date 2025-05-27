import anndata
import scanpy_modified as scanpy
from ripser import Rips
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from math import floor
from utils2 import normalize, density_downsampling, SEED
FEATURES_MAX_THRESHOLD = 5 #? 5 is given in the paper dc about logic
scanpy.settings.verbosity = 0


def get_geodestic_distance(da, neighbours_parameter:dict={}) -> np.ndarray:
    """
    Calculate the geodesic distance for the given data
    
    Args:
        df (np.ndarray): The data for which to calculate the geodesic distance.
        neighbours_parameter (dict): Parameters for the neighbors function.
    
    Returns:
        np.ndarray: The geodesic distance matrix.
    """
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=UserWarning)
    data = anndata.AnnData(da)
    data.uns['iroot'] = 0
    scanpy.pp.neighbors(data, **neighbours_parameter)
    scanpy.tl.diffmap(data)
    scanpy.tl.dpt(data)
    return np.stack(data.obs['dpt_distances'].to_list())

def metric_distribution_of_pairwise_distances(da, num_bins: int) -> np.number:
    """
    ref: https://doi.org/10.1371/journal.pcbi.1011866
    pg: 5
    title: Scoring metric 1 - distribution of pairwise distances
    
    Args:
        da (np.ndarrayy): Data array containing the data to be analyzed.
        num_bins (int): Number of bins to use for the histogram.
        
    Returns:
        np.number: The entropy of the histogram of pairwise distances.
    """
    size_da = len(da)
    tmp = get_geodestic_distance(da,{"n_neighbors":max(10,int(0.005*size_da)),"knn":True})
    tmp[tmp == np.inf] = 1.5 * np.max(tmp[tmp != np.inf]) 
    tmp[tmp == -np.inf] = -1 * np.min(tmp[tmp != -np.inf]) 
    a = plt.hist(tmp[np.triu(tmp, 1) != 0], bins = num_bins)
    hs = a[0]/np.sum(a[0])
   
    ent = entropy(hs, base=num_bins)
    assert type(ent) is np.float64, "pairewise_distance, return value is not a float or number"
    return  ent

def metric_persistent_homology(da, num_bins: int) -> np.number:
    """
    ref: https://doi.org/10.1371/journal.pcbi.1011866
    pg: 6
    title: Scoring metric 2 - persistent homology
    
    Args:
        da (np.ndarrayy): Data array containing the data to be analyzed.
        num_bins (int): Number of bins to use for the histogram.
        visualize (bool): If True, displays the histogram plot. Defaults to False.
        
    Returns:
        np.number: The entropy of the histogram of persistent homology distances.
    """   
    tmp = get_geodestic_distance(da)
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
    assert type(entropy_value) is np.float64, "entropy_value is not a float or number"
    return entropy_value

def metric_vector(da, cells_per_cluster:int=20, metric: str="euclidean") -> float | np.floating:
    """
    ref: https://doi.org/10.1371/journal.pcbi.1011866
    pg: 6
    title: Scoring metric 3 - vector magnitude
    
    Args:
        da (np.ndarrayy): Data array containing the data to be analyzed.
        cells_per_cluster (int, optional): Number of cells per cluster. Defaults to 20.
        metric (str, optional): Metric to use for clustering. Defaults to "euclidean".
        
    Returns:
        float | np.floating: The score calculated based on the clustering and vector analysis.
    """
    if da.shape[1] > FEATURES_MAX_THRESHOLD:
        da = PCA(n_components=FEATURES_MAX_THRESHOLD).fit_transform(da)
    number_of_clusters = min(floor(len(da)/cells_per_cluster), 100) #? 5% is given by the formula X/20 => 5% of the data
    score = 0
    REPETITIONS = 10
    for seed in range(REPETITIONS):
        np.random.seed(seed)
        kmeans = KMeans(n_clusters=number_of_clusters, random_state=seed).fit(da)
        clusters = kmeans.cluster_centers_
        all_clusters_idx = np.arange(len(clusters))
        all_dist = pairwise_distances(clusters , metric=metric)
        threshold = np.percentile(all_dist[np.triu(all_dist, 1) !=0], 20)
        for index in range(number_of_clusters):
            kmean_order = []
            pos_kmean_order = []
            current_idx = index
            for _ in range(len(clusters)):
                kmean_order.append(clusters[current_idx])
                pos_kmean_order.append(current_idx)
                
                dist_current = all_dist[current_idx]
                if 1 == len(all_clusters_idx)-len(kmean_order):
                    break
                next_index = get_next_index_not_in_kmean_order(dist_current, pos_kmean_order)
                if dist_current[next_index] > threshold:
                    break
                current_idx = next_index
            vectors = np.array([np.array(kmean_order[i+1]) - np.array(kmean_order[i]) for i in range(len(kmean_order)-1)])
            vectors_sum:np.ndarray = np.sum(vectors, axis=0)
            if not vectors_sum.all() == 0: #if the sum of the vectors is 0 then no need to calculate the norm  
                norm = np.linalg.norm(vectors_sum,ord=da.shape[1])
                score += norm
            
    return score/(REPETITIONS*number_of_clusters)

def get_next_index_not_in_kmean_order(dist_current: np.ndarray, pos_kmean_order: list[int]) -> int:
    """
    Get the next index in the distance array that is not in the kmean order
    
    Args:
        dist_current (np.ndarray): The current distance array.
        pos_kmean_order (list[int]): The list of indices that are already in the kmean order.
        
    Returns:
        int: The next index that is not in the kmean order.
    """
    closest_dist_index = 0
    sorted_dist = np.argsort(dist_current)
    next_index = sorted_dist[closest_dist_index]
    while next_index in pos_kmean_order:
        closest_dist_index += 1
        next_index = sorted_dist[closest_dist_index]
    return next_index

def metric_ripley_dpt(da, threshold: int = 100) ->  float | np.floating:
    """
    ref: https://doi.org/10.1371/journal.pcbi.1011866
    pg: 7
    title: Scoring method 4 - Ripley's k function

    Args:
        da (np.ndarrayy): Data array containing the data to be analyzed.
        threshold (int, optional): Threshold for the k function. Defaults to 100.

    Returns:
        float | np.floating: The Ripley score calculated from the k function.
    """
    n , nb_features = da.shape # "n" is the number of samples in the dataset
    dim_min = np.min(da, axis=0)
    dim_max = np.max(da, axis=0)
    ripley_score = 0
    REPEATITIONS = 1
    for _ in range(REPEATITIONS):
        bootstrap = np.random.uniform(dim_min, dim_max, (n, nb_features))

        geo_dist_df= get_geodestic_distance(da)
        geo_dist_bootstrap = get_geodestic_distance(bootstrap)

        geo_dist_df = adapt_inf(geo_dist_df)
        geo_dist_bootstrap = adapt_inf(geo_dist_bootstrap)

        k_geo_dist_df = k_function(geo_dist_df, threshold)
        k_geo_dist_bootstrap = k_function(geo_dist_bootstrap, threshold)

        sum_n = k_geo_dist_df + k_geo_dist_bootstrap
        k_geo_dist_df = k_geo_dist_df[sum_n != 2]
        k_geo_dist_bootstrap = k_geo_dist_bootstrap[sum_n != 2]
        
        local_score = np.trapezoid(np.abs(k_geo_dist_df - k_geo_dist_bootstrap), dx=1/len(k_geo_dist_df))
        ripley_score += local_score
    assert type(ripley_score) is np.float64, "ripley_score is not a float"
    return ripley_score / REPEATITIONS 
     
def adapt_inf(da: np.ndarray) -> np.ndarray:
    """
    Adapt the inf values in the ndarray to the max value time 1.5.
    
    Args:
        da (np.ndarray): The data array to adapt.
        
    Returns:
        np.ndarray: The adapted data array with inf values replaced.
    """
    da[da == np.inf] = 1.5 * np.max(da[da != np.inf])
    return da

def k_function(da: np.ndarray, threshold_size = 100) -> np.ndarray:
    """
    Calculate the k function for the given data.
    
    Args:
        df (np.ndarray): The data array for which to calculate the k function.
        threshold_size (int, optional): The size of the threshold. Defaults to 100.
    
    Returns:
        np.ndarray: The k function values for the data.
    """
    xs = np.linspace(0, np.nanmax(da[da != -np.inf])+1, threshold_size)
    k_value_ndarray = np.zeros(len(xs))
    for i in range(len(xs)):
        threshold = xs[i]
        k_value = np.sum(np.sum(da < threshold, axis=0) / da.shape[0]) # the number of samples is still the same as from the original data df/bootstrap (df.shape[0])
        k_value_ndarray[i] = k_value
    return normalize(k_value_ndarray) #the .values is to convert the dataframe to a numpy array

def metric_avg_connection(da) -> float | np.floating:
    """
    ref: https://doi.org/10.1371/journal.pcbi.1011866
    pg: 8
    title: Scoring metric 5 - degrees of connectivity
    
    Args:
        da (np.ndarrayy): Data array containing the data to be analyzed.
    
    Returns:
        float | np.floating: The average connection score calculated from the data.
    """
    c = density_downsampling(da, od=0.03, td=0.3)
    K = np.linspace(0.03,1,20) #* 5% to 95% of the number of data points. | shouldn't be linspace(0.05, 0.95, 19) 
    k_scores = []
    for k in K:
        score_k_dpt = generate_score_k_dpt(c, k) 
        k_scores.append(score_k_dpt)
    score = np.trapezoid(k_scores, K/np.max(K))
    assert type(score) is np.float64, "score is not a float"
    return score

def generate_score_k_dpt(da, k: float) -> float|np.floating:
    """
    This function generates the score for the k parameter based on the geodesic distance and the average connection.
    Args:
        da (np.ndarray): The data array for which to calculate the score.
        k (float): The k parameter to use for the score calculation.

    Returns:
        float | np.floating: The score calculated based on the geodesic distance and average connection.
    """
    np.random.seed(SEED)
    if len(da)>200:
        REPETITIONS = 5
        final_score = 0
        for _ in range(REPETITIONS):
            idx = np.random.randint(0, len(da), size=200)
            t_data = da[idx]
            geo_dist_df = get_geodestic_distance(t_data)
            geo_dist_df = adapt_inf(geo_dist_df)
            knn_distance_based = NearestNeighbors(n_neighbors=floor(len(t_data)*k), metric="precomputed").fit(geo_dist_df)
            
            Adjacency = knn_distance_based.kneighbors_graph(geo_dist_df).toarray() #type: ignore
            

            avg_connect = average_connection(Adjacency)
            final_score += np.median(avg_connect)
        return final_score / REPETITIONS
    else:
        geo_dist_df = get_geodestic_distance(da)
        geo_dist_df = adapt_inf(geo_dist_df)
        
        knn_distance_based: NearestNeighbors = NearestNeighbors(n_neighbors=max(1, floor(len(da)*k)), metric="precomputed").fit(geo_dist_df)
        
        Adjacency = knn_distance_based.kneighbors_graph(geo_dist_df).toarray() #type: ignore
        avg_connect = average_connection(Adjacency)
        score = np.median(avg_connect)
        return score

def average_connection(A: np.ndarray)-> np.ndarray:
    """
    This is a helper function to calculate the average connection
    
    Args:
        A (np.ndarray): The adjacency matrix of the graph.
        
    Returns:
        np.ndarray: The average connection for each node in the graph.
    """
    SA = ((A+A.T) > 1.5)*1
    old_total = 0
    total = np.sum(SA)
    loop = 0
    while total != old_total and (loop:=loop+1) <= 10:
        old_total = total
        SA = ((SA @ SA) > 1)*1
        total = np.sum(SA)
    return np.mean(SA, axis=0) 