import argparse
import numpy as np
import struct

'''
TODO:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Write a script to generate rimd features
    from .b files

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
def get_num_neighbors_from_file(header_file):
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

def parse_content(num_neighbors, file):
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

def rimd2feat(rimd_data):
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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type = str, default= './rimd-data/SMPL/')
    parser.add_argument('--dst_dir', type = str, default = './rimd-feature/SMPL/')
    opt = parser.parse_args()

    print ('[info] RIMD location: %s' % opt.src_dir)
    print ('[info] Target location: %s' % opt.dst_dir)

    num_neighbors = get_num_neighbors_from_file(opt.src_dir + 'header.b')

    num_models = 10000

    for i in range(num_models):
        file_name = opt.src_dir + str(i) + '.b'
        print ('Load %s' % file_name)
        rimd_data = parse_content(num_neighbors, file_name)
        rimd_feat = rimd2feat(rimd_data)
        np.save(opt.dst_dir + str(i), rimd_feat)
        print ('Save %s' % opt.dst_dir + str(i) + '.npy\n')

        # update bounds
        if i == 0:
            maxima = rimd_feat.copy()
            minima = rimd_feat.copy()

        else:
            for j in range(rimd_feat.shape[0]):

                if rimd_feat[j] > maxima[j]:
                    maxima[j] = rimd_feat[j].copy()
                if rimd_feat[j] < minima[j]:
                    minima[j] = rimd_feat[j].copy()

                if maxima[j] == minima[j]:
                    print ('The max value equals to the min value!')
                    maxima[j] = maxima[j] + 1e-6
                    minima[j] = minima[j] - 1e-6

    np.save(dst_dir + 'maxima', maxima)
    np.save(dst_dir + 'minima', minima)


if __name__ == '__main__':
    main()
