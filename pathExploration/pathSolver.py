import numpy as np
import torch
import higra
import bezier
import sympy
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('./training/')
from models import Encoder, Decoder
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz
from scipy.sparse.csgraph import dijkstra
from sklearn.decomposition import PCA


class PathSolver:

    def __init__(self, opt):
        self.opt = opt

        # Networks
        self.encoder = None
        self.decoder = None
        self.encoder_weights = './trained_weights/AAE_' + self.opt.dataset + '/Encoder.pth'
        self.decoder_weights = './trained_weights/AAE_' + self.opt.dataset + '/Decoder.pth'
        self.initEncoderDecoder()

        # KNN-MST Graph
        self.graph = None
        self.nodes_path = './pathExploration/KNN-MST/nodes_' + self.opt.dataset + '.npy'
        self.graph_path = './pathExploration/KNN-MST/graph_' + self.opt.dataset + '.npz'
        self.constructGraph()

        # Mesh Reconstruction
        self.minima = np.load('./ACAP-data/' + self.opt.dataset + '/minima.npy')
        self.maxima = np.load('./ACAP-data/' + self.opt.dataset + '/maxima.npy')

        self.seq_codes = None
        self.seq_ACAP = None

    # Initialize the network by loading training weights
    def initEncoderDecoder(self):

        if self.opt.dataset == 'SMPL':
            num_verts = 6890
        elif self.opt.dataset == 'all_animals':
            num_verts = 3889

        encoder = Encoder()
        decoder = Decoder(num_verts = num_verts)
        encoder.load_state_dict(torch.load(self.encoder_weights))
        decoder.load_state_dict(torch.load(self.decoder_weights))
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()


    # build KNN-MST graph for path exploration
    def constructGraph(self):

        # case 1： graph file exists
        if os.path.isfile(self.graph_path):
            print ('[info] Graph file exists.')
            self.graph = load_npz(self.graph_path)
            self.nodes = np.load(self.nodes_path)
        
        # case 2: graph file does not exist
        else:
            print ('[info] Graph file does not exist, trying to build the graph...')

            self.nodes = []
            for i in range(10000):
                x = np.load('./ACAP-data/' + self.opt.dataset + '/' + str(i) + '_norm.npy').astype(np.float32)
                x = torch.from_numpy(x)
                z = self.encoder(x.unsqueeze(0))
                self.nodes.append(z.squeeze(0).detach().numpy())
            self.nodes = np.array(self.nodes).astype(np.float32)

            np.save(self.nodes_path, self.nodes)

            n = self.nodes.shape[0]
            
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

            save_npz(self.graph_path, G)

            self.graph = G

    def back_mapping(self, feat):
        a = 0.95
        for i in range(feat.shape[0]):
            min_val = self.minima[i].copy()
            max_val = self.maxima[i].copy()
            feat[i] = ((feat[i] + a) / (2.0 * a)) * (max_val - min_val) + min_val

        return feat

    # Given the indices of source mesh and target mesh,
    # solve for a sequence of ACAP features
    def solve(self, s, t, mode = 'linear'):
        
        # step 1. solve for a sequence of latent codes
        if mode == 'linear':

            t_vals = np.linspace(0, 1, self.opt.num_samples)
            res = []
            for i in range(self.opt.num_samples):
                node = self.nodes[s] + (self.nodes[t] - self.nodes[s]) * t_vals[i]
                res.append(node)
            self.seq_codes = np.array(res)
            

        elif mode == 'bezier':
            # first solve for initial path
            dist_matrix, predecessors = dijkstra(csgraph=self.graph, directed=False, indices=s, return_predecessors=True)
            init_seq = []
            idx = t
            while idx >= 0:
                init_seq.append(idx)
                idx = predecessors[idx]
            
            init_seq.reverse()
            print ('[info] Initial path:', init_seq)
            init_seq = self.nodes[init_seq]
            
            # Then refine the path by Bezier curve
            control_nodes = init_seq.transpose()
            
            n_samples = self.opt.num_samples
            curve = bezier.Curve.from_nodes(control_nodes)
            
            
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
            interval = num_segments / self.opt.num_samples
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
            

            #t_vals = np.linspace(0, 1, self.opt.num_samples)

            samples = curve.evaluate_multi(t_vals)
            samples = samples.transpose()
            
            self.seq_codes = np.array(samples).astype(np.float32)

            arc_length = []
            for i in range(self.seq_codes.shape[0] - 1):
                dist = np.linalg.norm(self.seq_codes[i] - self.seq_codes[i + 1])
                arc_length.append(dist)
            arc_length = np.array(arc_length)

            print ('Standard deviation of arc lengths: ', arc_length.std())
            
        
        # step 2. reconstruct ACAP features from latent codes
        decoded = self.decoder(torch.from_numpy(self.seq_codes))
        self.seq_ACAP = []
        for item in decoded:
            self.seq_ACAP.append(self.back_mapping(item.detach().numpy()))
        self.seq_ACAP = np.array(self.seq_ACAP)

    # Visualize the explored path using PCA
    def show_path(self):

        # get PCA embedding and plot them
        graph_nodes = self.nodes
        curve_nodes = self.seq_codes
        z_dim = graph_nodes.shape[1]
        nodes = np.append(graph_nodes, curve_nodes).reshape(-1, z_dim)
        embedded = PCA(n_components=2).fit_transform(nodes)
        xs, ys = embedded.T
        plt.scatter(xs, ys, c = 'gray', alpha = 0.8)

        
        # plot bezier curve
        idx = graph_nodes.shape[0]
        plt.scatter(xs[idx:], ys[idx:], c = 'red')
        plt.plot(xs[idx:], ys[idx:], c = 'red', label = 'Bezier Curve')
        plt.xlabel('1st principal component')
        plt.ylabel('2nd principal component')
        plt.legend(loc='upper right')
        plt.show()