import numpy as np
import higra

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

n = 10000
nodes = np.random.randn(n, 2)

graph, weights = higra.make_graph_from_points(nodes, type = 'knn', n_neighbors = 1)


rows = []
cols = []
vals = []

for e in graph.edges():
    row, col, idx = e
    rows.append(row)
    cols.append(col)
    vals.append(weights[idx])


graph = csr_matrix((vals, (rows, cols)), shape = (n, n))

n_components, labels = connected_components(csgraph = graph, directed = False, return_labels = True)

print (n_components)