 
from sklearn.cluster import KMeans
from Chapter5.DistanceMetrics import InstanceDistanceMetrics
import sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from Chapter5.DistanceMetrics import PersonDistanceMetricsNoOrdering
from Chapter5.DistanceMetrics import PersonDistanceMetricsOrdering
import random
import scipy
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import DistanceMetric
import pyclus
