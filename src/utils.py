# Import necessary librariesps
import pandas as pd
import numpy as np
from scipy.spatial import distance

SEED = 42

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
    #? Computes condensed pairwise Euclidean distances between all rows (samples)
    #? Converts condensed vector into a square distance matrix of shape (n, n)
    #? dist_m[i][j] contains the Euclidean distance between point i and j
    dist = distance.pdist(df, metric='euclidean')
    dist_matrix = distance.squareform(dist)

    #? sort the the matrix based on the axis=1 (row-wise) thus each line (represents the distance between i and his closest neighbors)
    #? we take the median of the second closest neighbor (index 1) because the first one is the point itself (distance 0)
    sorted_dist_m = np.sort(dist_matrix)
    median_min_dist = np.median(sorted_dist_m[:,1])
    dist_threshold = np.max(median_min_dist)

    local_density = np.sum((dist_matrix < dist_threshold), axis=0) #! we removed the "1*" for converting to int

    od_threshold = np.quantile(local_density, od)
    td_threshold = np.quantile(local_density, td)
    
    np.random.seed(SEED)
    index_to_keep = []
    for i in range(len(local_density)):
        if local_density[i] < od_threshold:
            continue
        elif local_density[i] > td_threshold:
            if np.random.uniform(0,1) < td_threshold/local_density[i]:
                index_to_keep.append(i)
        else:
            index_to_keep.append(i)
    return df.iloc[index_to_keep] #! care we changed from df[index_to_keep,:] to df.iloc[index_to_keep]

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
    df = normalize(df)
    return df


if __name__ == "__main__":
    # test the functions
    df = pd.read_csv("/home/linsfa/Documents/BIO-INFO-2025/src/data/fig6_fig7/bone-marrow-mesenchyme-erythrocyte-differentiation_mca.rds.csv", index_col=0)
    print(preprocessing(df, 0.1, 0.9))