import numpy as np
from metrics import metric_distribution_of_pairwise_distances, metric_persistent_homology, metric_vector, metric_ripley_dpt, metric_avg_connection 
from utils2 import preprocessing

def scoring(df: np.ndarray, num_downsample= 5000):
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
    sc1 = metric_distribution_of_pairwise_distances(df, num_bins = 10)
    sc2 = metric_persistent_homology(df,num_bins = 3)
    sc3 = metric_vector(df)
    sc4 = metric_ripley_dpt(df)
    sc5 = metric_avg_connection(df)
    return [sc1,sc2,sc3,sc4,sc5]
    
    

if __name__ == '__main__':
    file = "data/fig6_fig7/bone-marrow-mesenchyme-erythrocyte-differentiation_mca.rds.csv"
   