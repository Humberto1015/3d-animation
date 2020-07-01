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

from mpl_toolkits.mplot3d import Axes3D

class Graph:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

        self.nodes = []
        self.edges = []
        self.dist = None

        self.source_idx = 300
        self.target_idx = 364

        self.loadData()
        self.buildConnectivity()

    def loadData(self):

        for i in range(0, 5000):
            x = np.load('./ACAP-data/SMPL/' + str(i) + '_norm.npy').astype(np.float32)
            x = torch.from_numpy(x)
            z = self.encoder(x.unsqueeze(0))
            self.nodes.append(z.squeeze(0).detach().numpy())
            
            # use fake data to debug first
            #self.nodes.append(np.random.randn(2))
        
        self.nodes = np.array(self.nodes)
        print (self.nodes.shape)
   
    def findConnectedComponents(self, graph):
        graph = csr_matrix(graph)
        n_components, labels = connected_components(csgraph = graph, directed = False, return_labels = True)
        print ('[info] Number of connected components: ', n_components)
        return n_components, labels
    
    def buildConnectivity(self):

        # precompute distances
        #n = self.nodes.shape[0]
        #nbrs = NearestNeighbors(n_neighbors = n, algorithm = 'ball_tree', metric = 'euclidean').fit(self.nodes)
        #dist, indices_all = nbrs.kneighbors(self.nodes)

        # step 1. build initial connectivity by finding k-nearest-neoghbors for each sample
        G = kneighbors_graph(self.nodes, 4, mode = 'distance', include_self = True)
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

    def solvePath(self):
        
        g = csr_matrix(self.graph)
        dist_matrix, predecessors = dijkstra(csgraph=g, directed=False, indices=0, return_predecessors=True)
        
        self.predecessors = predecessors
        #print (predecessors)

    def show(self):

        fig = plt.figure()
        ax = Axes3D(fig)
        
        pca = PCA(n_components = 3)
        embedded = pca.fit_transform(self.nodes)

        xs, ys, zs = embedded.T

        # plot nodes
        ax.scatter(xs, ys, zs, c = 'black')
        # plot edges
        #for edge in self.edges:
        #    s = edge[0]
        #    t = edge[1]
        #    plt.plot([xs[s], xs[t]], [ys[s], ys[t]], c = 'black', alpha = 0.25)

        # plot shortest path
        path = []
        idx = self.target_idx
        parent = self.predecessors[idx]
        while parent > 0:
            path.append([parent, idx])
            idx = parent
            parent = self.predecessors[idx]

        path.append([self.source_idx, idx])

        for pair in path:
            s = pair[0]
            t = pair[1]

            print (s, t)

            ax.scatter(xs[s], ys[s], zs[s], c = 'green')
            ax.scatter(xs[t], ys[t], zs[t], c = 'green')
            ax.plot([xs[s], xs[t]], [ys[s], ys[t]], [zs[s], zs[t]], c = 'green')

        #plt.scatter(xs[self.source_idx], ys[self.source_idx], c = 'r')
        #plt.scatter(xs[self.target_idx], ys[self.target_idx], c = 'r')

        plt.show()