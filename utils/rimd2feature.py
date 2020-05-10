"""
    a script that turns RIMD representation to training feature
"""
import os
import argparse
import struct
import numpy as np

class RIMDFeature:
    def __init__(self, src_dir, dst_dir):
        num_neighbors = self.get_num_neighbors_from_file(src_dir + 'header.b')

        # except for the header file
        self.num_models = len(os.listdir(src_dir)) - 1
        self.num_models = 5000

        for i in range(self.num_models):
            file_name = src_dir + str(i) + '.b'
            print ('Load %s' % file_name)
            rimd_data = self.parse_content(num_neighbors, file_name)
            rimd_feat = self.rimd2feat(rimd_data)
            np.save(dst_dir + str(i), rimd_feat)
            print ('Save %s' % dst_dir + str(i) + '.b\n')

            # update bounds
            if i == 0:
                self.maxima = rimd_feat.copy()
                self.minima = rimd_feat.copy()
            else:
                for j in range(rimd_feat.shape[0]):

                    if rimd_feat[j] > self.maxima[j]:
                        self.maxima[j] = rimd_feat[j].copy()
                    if rimd_feat[j] < self.minima[j]:
                        self.minima[j] = rimd_feat[j].copy()

        np.save(dst_dir + 'maxima', self.maxima)
        np.save(dst_dir + 'minima', self.minima)
    # get #of neighbors for each one ring center
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

    def parse_content(self, num_neighbors, file):
        # given number of neighbors, parse the one-ring information
        num_verts = num_neighbors.shape[0]

        fp = open(file, 'rb')
        buffer = fp.read()
        fp.close()

        RIMD = []

        index = 0
        # traverse all vertices
        for i in range(num_verts):

            one_ring = []
            # log_dRij matrix
            for j in range(num_neighbors[i]):
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

        return feat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type = str)
    parser.add_argument('--dst_dir', type = str)
    opt = parser.parse_args()

    # generate rimd feautre from rimd data
    data = RIMDFeature(opt.src_dir, opt.dst_dir)
    # normalize
