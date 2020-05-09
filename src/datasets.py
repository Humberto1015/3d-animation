from torch.utils.data import Dataset
import numpy as np
import struct
"""
[Binary representation]
    a.
        header.b:
        [n] [int] [int] [int] ... [int]

        where [n] denotes the number of vertices in the mesh.
            [int] denotes the neighbors number of each vertex.
    b.
        [mesh_idx].b
        [float] [float] [float] ... [float]
        [float] denotes the entry value of matrices.

With header.b, it is easy to parse the RIMD representations from binary files.
"""

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

class AnimalRIMD(Dataset):
    def __init__(self, train = True):
        self.train = train

        if self.train:
            self.path = './rimd-data/Animal_all/train/'
        else:
            self.path = './rimd-data/Animal_all/test/'

        self.num_neighbors = get_num_neighbors_from_file(self.path + 'header.b')

    def gen_feature(self, data):
        # generate the input feature from rimd vectors
        pass

    def __getitem__(self, index):
        data = parse_content(self.num_neighbors, self.path + str(index) + '.b')
        

    def __len__(self):
        if self.train:
            return 25000
        else:
            return 5000

def test_animal():
    dataset = AnimalRIMD(train = False)
    dataset.__getitem__(1)

if __name__ == '__main__':
    test_animal()
