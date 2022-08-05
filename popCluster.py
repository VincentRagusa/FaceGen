# based on code from https://gist.github.com/arthurratz/b7f788e46e924f463ad72842813e6fea#file-kmeans_pp-py
# See: https://towardsdatascience.com/k-means-algorithm-for-high-dimensional-data-clustering-714c6980daa9

#-----------------------------------------------------------------------------------
#   K-Means++ Optimal Data Clustering Algorithm v.0.0.1
#
#        C,S = compute(N,k)
#
#        N - # of observations, k - # of clusters
#
#        The worst-case complexity of the K-Means++ procedure:
#
#                   p = O(k^2ndi + nd)
#
#                   An Example: n = 10^2, d = 2, k = 3, i = 3
#
#                               p = O(3^2 * 10^2 * 2 * 3 + 10^2 * 2) = O(5400)
#
#   GNU Public License (C) 2021 Arthur V. Ratz
#-----------------------------------------------------------------------------------

import math
import numpy as np
from FaceGen import POP_SIZE, Org, loadData, loadPopulation, DATA_PATH, GENOME_LENGTH
import matplotlib.pyplot as plt


def exists(E,i):
    # return 'True' if the point i exists 
    # in the array E, and 'False' unless otherwise
    return 0 < np.shape(np.array( \
        [ e for e in np.array(E) if (e == i).all() ]))[0]

def eucld(i1,i2):
    # Compute the squared Euclidean distance d=|i1-i2|^2 as the sum of squared 
    # distances between points i1 and i2, at each dimension
    return np.sum(np.array([ \
        math.pow(i1 - i2, 2.0) \
          for i1, i2 in zip(i1, i2) ]))

def initialize(X):
    # Get the random centroid c0
    c0 = np.random.randint(0, np.shape(X)[0] - 1) + 1

    # Compute the distance from centroid c0 to each point in X
    c0_d = np.array([ eucld(X[c0], x) for x in X ])
    
    # Get the centroid c1's as one of the points in X,
    # having the maximum distance to the centroid c0
    c1 = np.where(c0_d >= np.max(c0_d))[0][0]
    
    return np.array([c0,c1]) # Return the indexes of c0 and c1

def compute(X,k):
    X = np.array(X)    # X - an input dataset of n-observations
    C = initialize(X)   # C - an initial set of centroids
    
    # Perform the dataset clustering iteratively, 
    # until the resultant set of k-clusters has been computed
    
    while True:
        S = np.empty(0)  # S - a set of newly built clusters
        
        # For each observation x[t] in X, do the following:
        for t in range(np.shape(X)[0]):
            # Check if the observation x[t] has already been
            # selected as one of the new centroids
            if exists(C, t) == False:
                # If not, compute the distance from 
                # the observation x[t] to each of the existing centroids in C
                cn_ds = np.array([ eucld(X[t], X[c]) for c in C ])
                # Get the centroid c[r] for which the distance to x[t] is the smallest
                cn_min_di = np.where(cn_ds == np.min(cn_ds))[0][0]

                # Assign the observation x[t] to the new cluster s[r], appending
                # the observation x[t]'s and centroid c[r]'s indexes to the set S
                S = np.append(S, { 'c': cn_min_di, 'i': t, 'd': cn_ds[cn_min_di] })

        # Terminate the clustering process, if the number of centroids 
        # in C is equal to the total number of clusters k, initially specified.
        
        # Otherwise, compute the next centroid c[r] in C
                
        if np.shape(C)[0] >= k: break
        
        # Get the distances |x-c| from the observations 
        # accross all clusters in S to each of the centroids in C
        cn_ds = np.array([s['d'] for s in S ])
        
        # Compute the index of an observation, for which 
        # the distance to one of the centroids in C is the largest
        cn_max_ci = np.where(cn_ds == np.max(cn_ds))[0][0]

        # Append the index of a new centroid c[r] to the set C
        C = np.append(C, S[cn_max_ci]['i'])

    return C,S

if __name__ == "__main__":
    NUM_CLUSTERS = 4
    U, S, Vh, SHAPE = loadData(DATA_PATH,GENOME_LENGTH)
    population = loadPopulation()

    # while True:
    centroids, assignments = compute([org.genome for org in population],NUM_CLUSTERS)
        # print( np.mean( [sum([1 for j in range(POP_SIZE-NUM_CLUSTERS) if assignments[j]["c"] == i]) + 1 for i in range(NUM_CLUSTERS)] ) )
        # if np.mean(np.mean( [sum([1 for j in range(POP_SIZE-NUM_CLUSTERS) if assignments[j]["c"] == i]) + 1 for i in range(NUM_CLUSTERS)] ) ) >= 0.8*(1/NUM_CLUSTERS)*POP_SIZE:
        #     break

    f, axarr = plt.subplots(1,NUM_CLUSTERS)
    f.set_size_inches((5*NUM_CLUSTERS,5))

    for i in range(NUM_CLUSTERS):
        clusterSet = [population[assignments[j]["i"]].genome for j in range(POP_SIZE-NUM_CLUSTERS) if assignments[j]["c"] == i] + [population[centroids[i]].genome]
        aveFace = np.mean(clusterSet, axis=0)
        img = np.reshape(np.array(np.clip(np.dot(aveFace, np.dot(np.diag(S[:GENOME_LENGTH]),Vh[:GENOME_LENGTH,:])),0,1)),SHAPE)
        axarr[i].imshow(img)
        axarr[i].set_title(len(clusterSet))
    plt.show()