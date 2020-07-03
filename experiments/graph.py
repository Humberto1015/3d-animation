import numpy as np
import torch

import matplotlib.pyplot as plt

from datasets import ACAPData

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.csgraph import connected_components

from sklearn.decomposition import PCA


class Graph:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

        self.nodes = []
        self.edges = []
        self.dist = None

        self.source_idx = 39
        self.target_idx = 364

        print ('[info] source index: %d, target index: %d' % (self.source_idx, self.target_idx))

        self.loadData()
        self.buildConnectivity()
        self.computeInitPath()
        self.optimizePath()

    def loadData(self):

        for i in range(0, 5000):
            x = np.load('./ACAP-data/SMPL/' + str(i) + '_norm.npy').astype(np.float32)
            x = torch.from_numpy(x)
            z = self.encoder(x.unsqueeze(0))
            self.nodes.append(z.squeeze(0).detach().numpy())
            
            # use fake data to debug first
            #self.nodes.append(np.random.randn(128).astype(np.float32))
        
        self.nodes = np.array(self.nodes)
    
    # given a graph, output:
    # (1) number of connected components
    # (2) a list showing labels
    def findConnectedComponents(self, graph):
        graph = csr_matrix(graph)
        n_components, labels = connected_components(csgraph = graph, directed = False, return_labels = True)
        print ('[info] Number of connected components: ', n_components)
        return n_components, labels
    
    # build the knn graph
    # make the graph connected if necessary (if there are more than 1 connected components)
    def buildConnectivity(self):

        # precompute distances
        #n = self.nodes.shape[0]
        #nbrs = NearestNeighbors(n_neighbors = n, algorithm = 'ball_tree', metric = 'euclidean').fit(self.nodes)
        #dist, indices_all = nbrs.kneighbors(self.nodes)

        # step 1. build initial connectivity by finding k-nearest-neoghbors for each sample
        G = kneighbors_graph(self.nodes, 6, mode = 'distance', include_self = True)
        G = G.toarray()

        self.graph = G
        
        for i in range(G.shape[0]):
            for j in range(G[i].shape[0]):
                if G[i][j] != 0:
                    self.edges.append([i, j])
        # step 2. find connected components of the graph
        n_components, labels = self.findConnectedComponents(G)
        
        # step 3. check if source sample and target smaple in the same connected component
        if not n_components == 1:
            print ('[info] The graph is not connected.')
            # do something here

        else:
            print ('[info] The graph is connected. [OK]')

    
    # compute the shortest path from source to target by dijkstra's algorithm
    # and save indices in member variable "self.seq"
    def computeInitPath(self):
        g = csr_matrix(self.graph)
        dist_matrix, predecessors = dijkstra(csgraph=g, directed=False, indices=self.source_idx, return_predecessors=True)
        self.seq = []
        idx = self.target_idx
        while idx > 0:
            self.seq.append(idx)
            idx = predecessors[idx]
        
        self.seq.reverse()
        print ('[info] Initial path:', self.seq)
        length, length_av = self.computeLength()
        print ('[info] Length = ', length)
        print ('[info] Average length = ', length_av)
    
    def computeLength(self):

        nodes = self.nodes[self.seq]
        length = 0
        for i in range(nodes.shape[0] - 1):
            length = length + np.linalg.norm(nodes[i] - nodes[i + 1])

        length_av = length / (nodes.shape[0] - 1)

        return length, length_av

    # Bezier Curve
    def one_bezier_curve(self, a,b,t):
        return (1-t)*a + t*b

    def n_bezier_curve(self, xs,n,k,t):
        if n == 1:
            return self.one_bezier_curve(xs[k],xs[k+1],t)
        else:
            return (1-t)*self.n_bezier_curve(xs,n-1,k,t) + t*self.n_bezier_curve(xs,n-1,k+1,t)

    # points: [dim, num_points]
    def bezier_curve(self, points, num_samples):
        
        n = points.shape[1] - 1
        dim = points.shape[0]
        t_step = 1.0 / (num_samples - 1)
        t = np.arange(0.0, 1 + t_step, t_step)

        res = np.zeros((dim, num_samples))
        
        for i in range(num_samples):
            for j in range(dim):
                res[j][i] = self.n_bezier_curve(points[j], n, 0, t[i])
        
        return res
    

    # Given the initial shortest path
    # optimize the path to make it as smooth as possible
    def optimizePath(self):
        
        print (self.seq)

        init_nodes = self.nodes[self.seq]
        init_nodes = init_nodes.transpose()
        
        n_samples = 30
        samples = self.bezier_curve(init_nodes, n_samples)
        samples = samples.transpose()
        
        self.solution = samples
        

    # Visualize the optimal path
    def show(self):
        pca = PCA(n_components = 2)
        embedded = pca.fit_transform(self.nodes)
        xs, ys = embedded.T
        plt.scatter(xs, ys, c = 'gray')

        plt.xlabel('1st principal component')
        plt.ylabel('2nd principal component')

        #n = len(self.seq)
        path_nodes = self.solution
        xs, ys = pca.transform(path_nodes).T
        
        # draw nodes
        plt.scatter(xs, ys, c = 'red')
        # draw edges
        plt.plot(xs, ys, c = 'red', label = 'initial path')
        #plt.arrow(nodes_x, nodes_y - nodes_x, c = 'green')
            
        # show linear interpolation path
        plt.plot([xs[0], xs[-1]], [ys[0], ys[-1]], c = 'blue')

        plt.show()