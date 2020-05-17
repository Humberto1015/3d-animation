import numpy as np
import mlrose
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

if __name__ == '__main__':

    """ Step 1. Sample key frames from N(0, I)  """
    mean = np.zeros(128)
    cov = np.eye(128)
    samples = np.random.multivariate_normal(mean, cov, 8)

    s = np.random.multivariate_normal(mean, cov, 1)
    t = np.random.multivariate_normal(mean, cov, 1)

    seq = samples.copy()
    seq = np.append(seq, s, axis = 0)
    seq = np.append(seq, t, axis = 0)
    """ ======================================== """

    """ == Step 2. Solve the TSP problem ======= """
    num_nodes = seq.shape[0]
    dist = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            d = np.linalg.norm(seq[i] - seq[j])
            dist.append((i, j, d))

    fitness_dists = mlrose.TravellingSales(distances = dist)
    problem_fit = mlrose.TSPOpt(length = num_nodes, fitness_fn = fitness_dists, maximize = False)
    best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 2)
    print (best_state)
    print (best_fitness)

    best_state = best_state.tolist()
    best_state.append(best_state[0])
    print (best_state)
    """ =========================================="""


    """ ========== Visualization ================ """
    # dimension reduction
    embedded = TSNE(n_components=2).fit_transform(seq)

    # visualization
    xs, ys = embedded.T
    # draw points
    plt.scatter(xs, ys)
    plt.scatter(xs[-1], ys[-1], c = 'r')
    plt.scatter(xs[-2], ys[-2], c = 'r')

    # draw edges
    for i in range(len(best_state) - 1):
        s = best_state[i]
        t = best_state[i + 1]
        x = [xs[s], xs[t]]
        y = [ys[s], ys[t]]
        plt.plot(x, y, c = 'black')

    plt.show()
    """ =========================================="""
