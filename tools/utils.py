import numpy as np
import trimesh
import torch
import sys
import visdom
import struct

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
