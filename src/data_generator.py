import numpy as np
import random
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import sklearn
import sklearn.metrics
from enum import Enum


class genType(Enum):
    RANDOM = "random",
    CLUSTER = "cluster", 
    TRAJECTORY = "trajectory"
    
    
def make_blob(num, mu_x, mu_y, sigma):
    #mu_, sigma = 0, 0.1 # mean and standard deviation
    X1 = np.random.normal(mu_x, sigma, num)
    Y1 = np.random.normal(mu_y, sigma, num)
    Blob = np.array([X1, Y1]).T
    return Blob

def make_blob_elipse(num, mu_x, mu_y, sigma_x, sigma_y):
    #mu_, sigma = 0, 0.1 # mean and standard deviation
    X1 = np.random.normal(mu_x, sigma_x, num)
    Y1 = np.random.normal(mu_y, sigma_y, num)
    Blob = np.array([X1, Y1]).T
    #plt.scatter(X1,Y1)
    return Blob

def gen_cluster(num=1000,seed=1):
    np.random.seed(seed)
    random.seed(seed)
    num_clusters = np.floor(np.random.exponential(scale=1.5)+2)
    prop = np.random.normal(5, 0.1, num_clusters) #SIGMA DETERMINES HOW UNBALANCED CLUSTERS ARE
    prop = prop/np.sum(prop)
    blobs = []
    center_cand_x = [i for i in range(num_clusters)]
    center_cand_y = [i for i in range(num_clusters)]
    for i in range(num_clusters):
        x= center_cand_x.pop(random.randrange(len(center_cand_x)))
        y= center_cand_y.pop(random.randrange(len(center_cand_y)))
        #print(x,y)
        blob1 = make_blob(int(num*prop[i]),x,y,np.random.normal(0.2,0.01))
        blobs.append(blob1)
    C = np.concatenate(blobs)
    return C


def gen_cluster_random(num=1000,seed= 1):
    np.random.seed(seed)
    random.seed(seed)
    num_clusters = np.random.randint(2,7)
    prop = np.random.uniform(0, 2, num_clusters) #SIGMA DETERMINES HOW UNBALANCED CLUSTERS ARE
    prop = prop/np.sum(prop)
    blobs = []
    center_cand_x = [i for i in range(num_clusters)]
    center_cand_y = [i for i in range(num_clusters)]
    for i in range(num_clusters):
        x= center_cand_x.pop(random.randrange(len(center_cand_x)))
        y= center_cand_y.pop(random.randrange(len(center_cand_y)))
        blob1 = make_blob_elipse(int(num*prop[i]),x,y,np.random.normal(0.3,0.02),np.random.normal(0.3,0.02))
        blobs.append(blob1)
    C = np.concatenate(blobs)
    return C


def gen_cluster_random_extreme(num=1000):
    num_clusters = np.random.randint(2,7)
    prop = np.random.uniform(0, 2, num_clusters) #SIGMA DETERMINES HOW UNBALANCED CLUSTERS ARE
    prop = prop/np.sum(prop)
    blobs = []
    center_cand_x = [i for i in range(num_clusters)]
    center_cand_y = [i for i in range(num_clusters)]
    xy = []
    for _ in range(num_clusters):
        x= center_cand_x.pop(random.randrange(len(center_cand_x)))
        y= center_cand_y.pop(random.randrange(len(center_cand_y)))
        xy.append([x,y])
    
    width = np.sqrt(np.max(sklearn.metrics.pairwise_distances(np.array(xy))))
    for i in range(len(xy)):
        x, y = xy[i]
        blob1 = make_blob_elipse(int(num*prop[i]),x,y,np.random.normal(width,width/10),np.random.normal(width,width/10))
        blobs.append(blob1)
        
    C = np.concatenate(blobs)
    return C

def gen_trajectory(num=1000, seed=1):
    np.random.seed(seed)
    random.seed(seed)
    if np.random.rand() < 0.5:
        X2 = np.random.uniform(0, np.random.uniform(0.5,3), num)
    else:
        center = np.random.uniform(3, 10)
        X2 = np.random.normal(center, 0.1*center, num)
    
    Y2 = np.sin(X2) + np.random.normal(0.15, np.random.uniform(0.01,0.05), num)
    C = np.array([X2, Y2]).T
    return C

def gen_trajectory_random(num=1000, seed=1):
    np.random.seed(seed)
    random.seed(seed)
    if np.random.rand() < 0.5:
        X2 = np.random.uniform(0, np.random.uniform(0.5,5), num)
        Y2 = np.sin(X2) + np.random.normal(0.5, np.random.uniform(0.05, 0.5), num)
        C = np.array([X2, Y2]).T
        return C
    else:
        width = max(np.random.normal(0.25, 0.05), 0) #* Ensure width is non-negative but didnt want to use the .2 from the original code
        down = np.random.randint(1, 3)
        up = np.random.randint(3, 5)
        X2 = np.linspace(0, down , num=int(num/3))
        Y2 = X2 + np.random.normal(0, width, int(num/3))
        
        X_bifurcation = np.linspace(down, up, num=int(num/3))
        first_slope = np.random.uniform(-5, 5)
        second_slope = np.random.uniform(-5, 5)
        Y_first_bifurcation = first_slope*X2 + np.random.normal(0, width, int(num/3)) + Y2[-1] #! the Y2[-1] is there for continuity of the X2 trajectory this is also why we can use X2 in the equation of slope 
        Y_second_bifurcation = second_slope*X2 + np.random.normal(0, width, int(num/3)) + Y2[-1]
        
        C = np.concatenate([np.array([X2,Y2]).T, np.array([X_bifurcation,Y_first_bifurcation]).T, np.array([X_bifurcation,Y_second_bifurcation]).T])
        return C
    
def gen_trajectory_random_extreme(num=1000):
    X2 = np.random.uniform(0, np.random.uniform(0.5,5), num)
    width = max(X2)-min(X2)
    Y2 = np.sin(X2) + np.random.normal(width, width, num)
    C = np.array([X2,Y2]).T
    return C

def gen_random(num=1000, mode=genType.RANDOM, seed=1):
    np.random.seed(seed)
    random.seed(seed)
    match mode:
        case genType.RANDOM: 
            ind = random.choice([0,1])
            model_chosen= [gen_cluster_random,gen_trajectory_random][ind]
        case genType.CLUSTER: 
            model_chosen = gen_cluster_random
            ind = 0
        case genType.TRAJECTORY: 
            model_chosen = gen_trajectory_random
            ind = 1
        case _:
            raise ValueError
    param = {}
    param['num'] = num 
    param["seed"] = seed
    C = model_chosen(**param)
    return C, ind