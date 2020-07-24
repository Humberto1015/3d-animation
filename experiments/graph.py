import numpy as np
import torch
import bezier
import matplotlib.pyplot as plt
import sympy

from datasets import ACAPData

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.csgraph import connected_components

import higra

from sklearn.decomposition import PCA

from timer import Timer

class Graph:
    def __init__(self, nodes):
        
        self.timer = Timer()


        self.nodes = nodes
        self.edges = []

        self.source_idx = None
        self.target_idx = None
        self.num_sample_nodes = 60

        self.timer.start()
        self.buildConnectivity()
        self.timer.end()

        print ('It cost %f sec to build graph connectivity.' % self.timer.time)

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
        n = self.nodes.shape[0]
        # step 1. build the knn-mst graph

        # try to implement it without calling higra functions
        G, weights = higra.make_graph_from_points(self.nodes, type = 'knn+mst', n_neighbors = 6)

        # convert the graph to sparse matrix
        rows = []
        cols = []
        vals = []
        for e in G.edges():
            row, col, idx = e
            rows.append(row)
            cols.append(col)
            vals.append(weights[idx])

        rows = np.array(rows)
        cols = np.array(cols)
        vals = np.array(vals)
        G = csr_matrix((vals, (rows, cols)), shape = (n, n))

        rows, cols = G.nonzero()
        self.graph = G

        for i in range(rows.shape[0]):
            self.edges.append([rows[i], cols[i]])

        # step 2. find connected components of the graph
        n_components, self.labels = self.findConnectedComponents(G)
        
        # step 3. check if source sample and target smaple in the same connected component
        if not n_components == 1:
            print ('[info] The graph is not connected.')

        else:
            print ('[info] The graph is connected. [OK]')

    
    # compute the shortest path from source to target by dijkstra's algorithm
    # and save indices in member variable "self.seq"
    def computeInitPath(self):
        g = self.graph
        dist_matrix, predecessors = dijkstra(csgraph=g, directed=False, indices=self.source_idx, return_predecessors=True)
        self.seq = []
        idx = self.target_idx
        while idx >= 0:
            self.seq.append(idx)
            idx = predecessors[idx]
        
        self.seq.reverse()
        self.init_path = self.nodes[self.seq]
        print ('[info] Initial path:', self.seq)
    
    # Given the initial shortest path
    # optimize the path to make it as smooth as possible
    def optimizePath(self):
        
        init_nodes = self.nodes[self.seq]
        init_nodes = init_nodes.transpose()
        
        n_samples = self.num_sample_nodes
        curve = bezier.Curve.from_nodes(init_nodes)

        # option 1. With uniform speed
        # f(t)
        f = curve.to_symbolic()
            
        # df/dt
        gradient = []
        for i in range(128):
            s = sympy.Symbol('s')
            gradient.append(f[i].diff(s))
        V = []
            
        level = len(sympy.Poly(gradient[0]).coeffs())
            
        for i in range(level):
            v = np.zeros(128)
            for j in range(128):
                v[j] = sympy.Poly(gradient[j]).coeffs()[i]
            V.append(v)
            
        V = np.array(V)

        num_segments = 1200
        t_vals = []
        t = 0
        interval = num_segments / self.num_sample_nodes
        for i in range(num_segments):
            if i % interval == 0:
                t_vals.append(t)
                
            step = 0
            for j in range(V.shape[0]):
                step = step + (t**(V.shape[0] - 1 - j)) * V[j]
            step = np.linalg.norm(step)
            t = t + (curve.length / num_segments) / step
            
        t_vals.append(1.0)
        t_vals = np.array(t_vals)

       
        

        samples = curve.evaluate_multi(t_vals)
        samples = samples.transpose()
        
        self.solution = samples
        

    def solve(self, s, t):
        
        self.source_idx = s
        self.target_idx = t

        

        self.timer.start()
        self.computeInitPath()
        self.timer.end()

        print ('It cost %f sec to compute initial path.' % self.timer.time)

        self.timer.start()
        self.optimizePath()
        self.timer.end()

        print ('It cost %f sec to optimize the path.' % self.timer.time)

    # Visualize the optimal path
    def show(self):

        # get PCA embedding and plot them
        graph_nodes = self.nodes
        curve_nodes = self.solution
        z_dim = graph_nodes.shape[1]
        nodes = np.append(graph_nodes, curve_nodes).reshape(-1, z_dim)
        pca = PCA(n_components = 2)
        embedded = pca.fit_transform(nodes)
        xs, ys = embedded.T
        plt.scatter(xs, ys, c = 'gray', alpha = 0.8)

        # plot shortest path obtained by Dijkstra's algorithm
        #xs_init, ys_init = pca.transform(self.init_path).T
        #plt.scatter(xs_init, ys_init, c = 'black')
        #plt.plot(xs_init, ys_init, c = 'black', label = 'Shortest Path')


        # plot bezier curve
        idx = graph_nodes.shape[0]
        plt.scatter(xs[idx:], ys[idx:], c = 'red')
        plt.plot(xs[idx:], ys[idx:], c = 'red', label = 'Bezier Curve')

        # show linear interpolation path
        source = embedded[idx]
        target = embedded[-1]
        t_vals = np.linspace(0, 1, 31)

        nodes = []
        for t in t_vals:
            nodes.append(source + t * (target - source))
        nodes = np.array(nodes)

        xs, ys = nodes.T

        plt.scatter(xs, ys, c = 'blue')
        plt.plot(xs, ys, c = 'blue', label = 'Linear Interpolation')

        plt.xlabel('1st principal component')
        plt.ylabel('2nd principal component')
        plt.legend(loc='upper right')
        plt.show()