import struct
import numpy as np

class RIMDData:
    def __init__(self, header_path, minima_path, maxima_path):
        self.num_neighbors = self.getNumOfNeighbors(header_path)

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

    def backMapping(self, val, min_val, max_val, a = 0.9):
        return ((val + a) / (2.0 * a)) * (max_val - min_val) + min_val
