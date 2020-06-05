import numpy as np
import trimesh
import torch
import sys
import visdom
import struct
import argparse


class Converter:
    def __init__(self, opt):

        self.num_neighbors = self.get_num_neighbors_from_file(opt.header_path)
        self.minima = np.load(opt.minima_path)
        self.maxima = np.load(opt.maxima_path)

    def get_num_neighbors_from_file(self, header_file):
        # Read the header file to get neighbors number of each vertex
        num_neighbors = []
        fp = open(header_file, 'rb')
        buffer = fp.read()
        fp.close()
        index = 0
        num_verts, = struct.unpack_from('<i', buffer, index)
        index = index + struct.calcsize('<i')

        for i in range(num_verts):
            value, = struct.unpack_from('<i', buffer, index)
            index = index + struct.calcsize('<i')
            num_neighbors.append(value)

        return np.array(num_neighbors)

    def get_RIMD_from_file(self, file):
        # given number of neighbors, parse the one-ring information
        num_verts = self.num_neighbors.shape[0]

        fp = open(file, 'rb')
        buffer = fp.read()
        fp.close()

        RIMD = []

        index = 0
        # traverse all vertices
        for i in range(num_verts):

            one_ring = []
            # log_dRij matrix
            for j in range(self.num_neighbors[i]):
                m = np.zeros((3, 3))
                for k in range(3):
                    for l in range(3):
                        value,  = struct.unpack_from('<f', buffer, index)
                        index = index + struct.calcsize('<f')
                        m[k][l] = value

                one_ring.append(m)

            # Si matrix
            m = np.zeros((3, 3))
            for j in range(3):
                for k in range(3):
                    value, = struct.unpack_from('<f', buffer, index)
                    index += struct.calcsize('<f')
                    m[j][k] = value
            one_ring.append(m)
            RIMD.append(one_ring)

        return RIMD

    def backMapping(self, val, min_val, max_val, a = 0.95):
        return ((val + a) / (2.0 * a)) * (max_val - min_val) + min_val

    def normalize(self, feat, a = 0.95):
        num_dim = feat.shape[0]

        for i in range(num_dim):
            min_val = self.minima[i].copy()
            max_val = self.maxima[i].copy()

            feat[i] = 2 * a * ((feat[i] - min_val)/ (max_val - min_val)) -a
        return feat

    def rimd2feat(self, rimd_data):
        feat = []
        for i in range(len(rimd_data)):
            vector = []
            matrices = rimd_data[i]
            # filling theta_x, theta_y, and theta_z
            for k in range(len(matrices) - 1):
                vector.append(matrices[k][2][1]) # theta_x
                vector.append(matrices[k][0][2]) # theta_y
                vector.append(matrices[k][1][0]) # theta_z
            # filling elements of scaling/shear matrix
            vector.append(matrices[-1][0][0])
            vector.append(matrices[-1][0][1])
            vector.append(matrices[-1][0][2])
            vector.append(matrices[-1][1][1])
            vector.append(matrices[-1][1][2])
            vector.append(matrices[-1][2][2])
            feat.extend(vector)
        feat = np.array(feat)

        return self.normalize(feat)

    def feat2RIMD(self, feat):

        num_verts = len(self.num_neighbors)
        rimd = []
        for i in range(num_verts):
            rimd.append([])

        one_ring_index = 0
        feat_index = 0
        for nb_neighbors in self.num_neighbors:


            for i in range(nb_neighbors):
                logdRij = np.zeros((3, 3))
                # theta_x
                min_val = self.minima[feat_index]
                max_val = self.maxima[feat_index]
                theta_x = self.backMapping(feat[feat_index], min_val, max_val, a=0.95)
                logdRij[1][2] = -theta_x
                logdRij[2][1] = theta_x
                feat_index += 1
                # theta_y
                min_val = self.minima[feat_index]
                max_val = self.maxima[feat_index]
                theta_y = self.backMapping(feat[feat_index], min_val, max_val, a=0.95)
                logdRij[0][2] = theta_y
                logdRij[2][0] = -theta_y
                feat_index += 1
                # theta_z
                min_val = self.minima[feat_index]
                max_val = self.maxima[feat_index]
                theta_z = self.backMapping(feat[feat_index], min_val, max_val, a=0.95)
                logdRij[0][1] = -theta_z
                logdRij[1][0] = theta_z
                feat_index += 1

                rimd[one_ring_index].append(logdRij)

            Si = np.zeros((3, 3))
            min_val = self.minima[feat_index]
            max_val = self.maxima[feat_index]
            Si[0][0] = self.backMapping(feat[feat_index], min_val, max_val, a=0.95)
            feat_index += 1
            min_val = self.minima[feat_index]
            max_val = self.maxima[feat_index]
            Si[0][1] = self.backMapping(feat[feat_index], min_val, max_val, a=0.95)
            Si[1][0] = self.backMapping(feat[feat_index], min_val, max_val, a=0.95)
            feat_index += 1
            min_val = self.minima[feat_index]
            max_val = self.maxima[feat_index]
            Si[0][2] = self.backMapping(feat[feat_index], min_val, max_val, a=0.95)
            Si[2][0] = self.backMapping(feat[feat_index], min_val, max_val, a=0.95)
            feat_index += 1
            min_val = self.minima[feat_index]
            max_val = self.maxima[feat_index]
            Si[1][1] = self.backMapping(feat[feat_index], min_val, max_val, a=0.95)
            feat_index += 1
            min_val = self.minima[feat_index]
            max_val = self.maxima[feat_index]
            Si[1][2] = self.backMapping(feat[feat_index], min_val, max_val, a=0.95)
            Si[2][1] = self.backMapping(feat[feat_index], min_val, max_val, a=0.95)
            feat_index += 1
            min_val = self.minima[feat_index]
            max_val = self.maxima[feat_index]
            Si[2][2] = self.backMapping(feat[feat_index], min_val, max_val, a=0.95)
            feat_index += 1

            rimd[one_ring_index].append(Si)

            one_ring_index += 1

        return rimd

    def rimd2file(self, target_path, rimd_data):

        fout = open(target_path, 'wb')

        for i in range(len(rimd_data)):

            one_ring = rimd_data[i]

            # dRij
            for j in range(len(one_ring) - 1):
                for k in range(3):
                    for l in range(3):
                        fout.write(struct.pack('<f', one_ring[j][k][l]))
            # Si
            for j in range(3):
                for k in range(3):
                    fout.write(struct.pack('<f', one_ring[-1][j][k]))
        fout.close()

class Visualizer:
    def __init__(self, env, port = 8888):
        self._vis = visdom.Visdom(port = port, env = env)

    def draw_2d_points(self, win, verts, color):
        self._vis.scatter(
            X = verts,
            win = win,
            update = 'append',
            opts = {
                'markersize': 5,
                'markercolor': color,
                'layoutopts': {
                    'plotly': {
                        'xaxis': {
                            'range': [-50, 50],
                            'autorange': False,
                        },
                        'yaxis': {
                            'range': [-50, 50],
                            'autorange': False,
                        },
                    }
                }
            }
        )
    def draw_line(self, win, x, y):
        self._vis.line(
            X = torch.Tensor([x]),
            Y = torch.Tensor([y]),
            win = win,
            update = 'append' if x > 0 else None,
            opts = {
                'title': win
            }
        )

def debug():

    print ('[Debug information]')
    parser = argparse.ArgumentParser()
    parser.add_argument('--header_path', type = str, default = './rimd-data/SMPL/header.b')
    parser.add_argument('--minima_path', type = str, default = './rimd-feature/SMPL/minima.npy')
    parser.add_argument('--maxima_path', type = str, default = './rimd-feature/SMPL/maxima.npy')
    opt = parser.parse_args()

    converter = Converter(opt)
    print ('Shape of minima array: ', converter.minima.shape)
    print ('Shape of maxima array: ', converter.maxima.shape)

    rimd_path = './rimd-data/SMPL/' + str(198) + '.b'
    rimd = converter.get_RIMD_from_file(rimd_path)
    feat = converter.rimd2feat(rimd)
    rimd = converter.feat2RIMD(feat)
    converter.rimd2file(str(198) + '.b', rimd)


if __name__ == '__main__':
    debug()
