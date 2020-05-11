import struct
import numpy as np
import argparse

def write2file(file_name, rimd_data):
    fout = open(file_name, 'wb')

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

class RIMDTransformer:
    def __init__(self, header_path, minima_path, maxima_path):
        self.num_neighbors = self.getNumOfNeighbors(header_path)
        self.minima = np.load(minima_path)
        self.maxima = np.load(maxima_path)

    def getNumOfNeighbors(self, header_path):
        num_neighbors = []
        fp = open(header_path, 'rb')
        buffer = fp.read()
        fp.close()
        index = 0
        num_verts, = struct.unpack_from('<i', buffer, index)
        index += struct.calcsize('<i')

        # get the 1-ring information
        for i in range(num_verts):
            value, = struct.unpack_from('<i', buffer, index)
            index += struct.calcsize('<i')
            num_neighbors.append(value)

        return num_neighbors

    # given a feature, turn it to RIMD format
    def turn2RIMD(self, feat):

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
                theta_x = self.backMapping(feat[feat_index], min_val, max_val, a=0.9)
                logdRij[1][2] = -theta_x
                logdRij[2][1] = theta_x
                feat_index += 1
                # theta_y
                min_val = self.minima[feat_index]
                max_val = self.maxima[feat_index]
                theta_y = self.backMapping(feat[feat_index], min_val, max_val, a=0.9)
                logdRij[0][2] = theta_y
                logdRij[2][0] = -theta_y
                feat_index += 1
                # theta_z
                min_val = self.minima[feat_index]
                max_val = self.maxima[feat_index]
                theta_z = self.backMapping(feat[feat_index], min_val, max_val, a=0.9)
                logdRij[0][1] = -theta_z
                logdRij[1][0] = theta_z
                feat_index += 1

                rimd[one_ring_index].append(logdRij)

            Si = np.zeros((3, 3))
            min_val = self.minima[feat_index]
            max_val = self.maxima[feat_index]
            Si[0][0] = self.backMapping(feat[feat_index], min_val, max_val, a=0.9)
            feat_index += 1
            min_val = self.minima[feat_index]
            max_val = self.maxima[feat_index]
            Si[0][1] = self.backMapping(feat[feat_index], min_val, max_val, a=0.9)
            Si[1][0] = self.backMapping(feat[feat_index], min_val, max_val, a=0.9)
            feat_index += 1
            min_val = self.minima[feat_index]
            max_val = self.maxima[feat_index]
            Si[0][2] = self.backMapping(feat[feat_index], min_val, max_val, a=0.9)
            Si[2][0] = self.backMapping(feat[feat_index], min_val, max_val, a=0.9)
            feat_index += 1
            min_val = self.minima[feat_index]
            max_val = self.maxima[feat_index]
            Si[1][1] = self.backMapping(feat[feat_index], min_val, max_val, a=0.9)
            feat_index += 1
            min_val = self.minima[feat_index]
            max_val = self.maxima[feat_index]
            Si[1][2] = self.backMapping(feat[feat_index], min_val, max_val, a=0.9)
            Si[2][1] = self.backMapping(feat[feat_index], min_val, max_val, a=0.9)
            feat_index += 1
            min_val = self.minima[feat_index]
            max_val = self.maxima[feat_index]
            Si[2][2] = self.backMapping(feat[feat_index], min_val, max_val, a=0.9)
            feat_index += 1

            rimd[one_ring_index].append(Si)

            one_ring_index += 1

        return rimd

    def backMapping(self, val, min_val, max_val, a = 0.9):
        return ((val + a) / (2.0 * a)) * (max_val - min_val) + min_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--header_path', type = str)
    parser.add_argument('--minima_path', type = str)
    parser.add_argument('--maxima_path', type = str)
    opt = parser.parse_args()

    header_path = './rimd-data/Animal_all/test/header.b'
    minima_path = './rimd-feature/Animal_all/test/minima.npy'
    maxima_path = './rimd-feature/Animal_all/test/maxima.npy'

    T = RIMDTransformer(header_path, minima_path, maxima_path)

    feature_path = './rimd-feature/Animal_all/test/'

    for i in range(10):
        feat = np.load(feature_path + str(i) + '_norm.npy')
        rimd = T.turn2RIMD(feat)

        #print (rimd)


        write2file(str(i) + '.b', rimd)
