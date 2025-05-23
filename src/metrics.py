import warnings
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

def metric_distribution_of_pairwise_distances(df, num_bins: int, visualize: bool = False) -> np.number:
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
    tmp = np.stack(df.obs['dpt_distances'].to_list()) 
    tmp[tmp == np.inf] = 1.5 * np.max(tmp[tmp != np.inf]) 
    tmp[tmp == -np.inf] = -1 * np.min(tmp[tmp != -np.inf]) 
    a = plt.hist(tmp[np.triu(tmp, 1) != 0], bins = num_bins)
    hs = a[0]/np.sum(a[0])
   
    ent = entropy(hs, base=num_bins)
    if visualize:
        plt.show()
    else:
        plt.close()
    assert type(ent) is np.float64, "pairewise_distance, return value is not a float or number"
    return  ent

def metric_persistent_homology(df, num_bins: int, visualize: bool = False) -> np.number:
    """
    ref: https://doi.org/10.1371/journal.pcbi.1011866
    pg: 6
    title: Scoring metric 2 -persistent homology
    """   
    df = anndata.AnnData(df)
    df.uns['iroot'] = 0
    scanpy.pp.neighbors(df)
    scanpy.tl.diffmap(df)
    scanpy.tl.dpt(df)
    tmp = np.stack(df.obs['dpt_distances'].to_list()) 
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
    assert type(entropy_value) is np.float64, "entropy_value is not a float or number"
    return entropy_value

def metric_vector(df, cells_per_cluster:int=20, metric: str="euclidean") -> float | np.floating:
    if df.shape[1] > FEATURES_MAX_THRESHOLD:
        df = PCA(n_components=FEATURES_MAX_THRESHOLD).fit_transform(df)
    number_of_clusters = min(floor(len(df)/cells_per_cluster), 100) #? 5% is given by the formula X/20 => 5% of the data
    score = 0
    REPETITIONS = 10
    for seed in range(REPETITIONS):
        np.random.seed(seed)
        kmeans = KMeans(n_clusters=number_of_clusters, random_state=seed).fit(df)
        for index in range(number_of_clusters):
            clusters = kmeans.cluster_centers_.tolist()
            all_dist = pairwise_distances(clusters , metric=metric)
            threshold = np.percentile(all_dist[np.triu(all_dist, 1) !=0], 20)
            current_idx = index
            kmean_order = []
            for _ in range(len(clusters)):
                kmean_order.append(clusters.pop(current_idx))
                dist_current = pairwise_distances(np.array(kmean_order[-1]).reshape(1,-1), clusters, metric=metric)[0] #todo later
                if len(dist_current) == 1:
                    break
                next_index = np.argsort(dist_current)[0]  
                if dist_current[next_index] > threshold:
                    break
                current_idx = next_index
            vectors = []
            for j in range(len(kmean_order)-1):
                d1 = np.array(kmean_order[j])
                d2 = np.array(kmean_order[j+1])
                vectors.append(d2-d1)       
            vectors_sum:np.ndarray = np.sum(vectors, axis=0)
            if not vectors_sum.all() == 0: #if the sum of the vectors is 0 then no need to calculate the norm  
                norm = np.linalg.norm(vectors_sum,ord=df.shape[1])
                score += norm
            
    return score/(REPETITIONS*number_of_clusters)

def metric_ripley_dpt(df, threshold: int = 100, visualize: bool = False) ->  float | np.floating:
    """_summary_

    Args:
        df (_type_): _description_
        threshold (int, optional): _description_. Defaults to 100.
        visualize (bool, optional): _description_. Defaults to False.

    Returns:
        float | np.floating: _description_
    """
    # "n" is the number of samples in the dataset
    n , nb_features = df.shape
    dim_min = np.min(df, axis=0)
    dim_max = np.max(df, axis=0)
    ripley_score = 0
    REPEATITIONS = 1
    for _ in range(REPEATITIONS):
        bootstrap = np.random.uniform(dim_min, dim_max, (n, nb_features))

        geo_dist_df= get_geodestic_distance(df, {})
        geo_dist_bootstrap = get_geodestic_distance(bootstrap, {})

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

def get_geodestic_distance(df, neighbours_parameter:dict) -> np.ndarray:
    """
    Calculate the geodesic distance for the given data
    """
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=UserWarning)
    data = anndata.AnnData(df)
    data.uns['iroot'] = 0
    scanpy.pp.pca(data)
    scanpy.pp.neighbors(data, **neighbours_parameter, use_rep='X_pca')
    scanpy.tl.diffmap(data)
    scanpy.tl.dpt(data)
    return np.stack(data.obs['dpt_distances'].to_list())
     
def adapt_inf(df: np.ndarray) -> np.ndarray:
    """
    Adapt the inf values in the ndarray to the max value time 1.5
    """
    df[df == np.inf] = 1.5 * np.max(df[df != np.inf])
    return df

def k_function(df: np.ndarray, threshold_size = 100) -> np.ndarray:
    """
    Calculate the k function for the given data
    """
    xs = np.linspace(0, np.nanmax(df[df != -np.inf])+1, threshold_size)
    k_value_ndarray = np.zeros(len(xs))
    for i in range(len(xs)):
        threshold = xs[i]
        k_value = np.sum(np.sum(df < threshold, axis=0) / df.shape[0]) # the number of samples is still the same as from the original data df/bootstrap (df.shape[0])
        k_value_ndarray[i] = k_value
    return normalize(k_value_ndarray) #the .values is to convert the dataframe to a numpy array

def metric_avg_connection(df) -> float | np.floating:
    """
    This is the metric 5 also known as the "degrees of connectivity" metric.    
    """
    c = density_downsampling(df, od=0.03, td=0.3)
    K = np.linspace(0.03,1,20) #* 5% to 95% of the number of data points. | shouldn't be linspace(0.05, 0.95, 19) 
    k_scores = []
    for k in K:
        score_k_dpt = generate_score_k_dpt(c, k) 
        k_scores.append(score_k_dpt)
    score = np.trapezoid(k_scores, K/np.max(K))
    assert type(score) is np.float64, "score is not a float"
    return score

def generate_score_k_dpt(df, k: float) -> float|np.floating:
    """

    Args:
        df (_type_): _description_
        k (float): _description_

    Returns:
        float|np.floating: _description_
    """
    np.random.seed(SEED)
    if len(df)>200:
        REPETITIONS = 5
        final_score = 0
        for _ in range(REPETITIONS):
            idx = np.random.randint(0, len(df), size=200)
            t_data = df[idx]
            geo_dist_df = get_geodestic_distance(t_data, {})
            geo_dist_df = adapt_inf(geo_dist_df)
            knn_distance_based = NearestNeighbors(n_neighbors=floor(len(t_data)*k), metric="precomputed").fit(geo_dist_df)
            
            Adjacency = knn_distance_based.kneighbors_graph(geo_dist_df).toarray() #type: ignore
            

            avg_connect = average_connection(Adjacency)
            final_score += np.median(avg_connect)
        return final_score / REPETITIONS
    else:
        geo_dist_df = get_geodestic_distance(df, {})
        geo_dist_df = adapt_inf(geo_dist_df)
        
        knn_distance_based: NearestNeighbors = NearestNeighbors(n_neighbors=max(1, floor(len(df)*k)), metric="precomputed").fit(geo_dist_df)
        
        Adjacency = knn_distance_based.kneighbors_graph(geo_dist_df).toarray() #type: ignore
        avg_connect = average_connection(Adjacency)
        score = np.median(avg_connect)
        return score

def average_connection(A: np.ndarray)-> np.ndarray:
    """
    This is a helper function to calculate the average connection
    
    Args:
        A (np.ndarray): is the Adjacency matrix of the graph
        
    Returns:
        np.ndarray: the average connection of the graph 
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


if __name__ == "__main__":
    # Example usage
    from utils2 import preprocessing
    # path = "/home/linsfa/Documents/BIO-INFO-2025/src/data/fig6_fig7/"
    # for file in os.listdir(path):
    #     if file.endswith(".rds.csv"):
    #         print(file)
    #         df = pd.read_csv(path + file, index_col=0)
    #         df = preprocessing(df, 0.1, 0.9)
    #         metric_persistent_homology(df, num_bins=10)
    path = "/home/linsfa/Documents/BIO-INFO-2025/src/data/fig6_fig7/bone-marrow-mesenchyme-erythrocyte-differentiation_mca.rds.csv"
    df = pd.read_csv(path, index_col=0)
    # df = preprocessing(df, 0.1, 0.9)
    print(metric_avg_connection(df))