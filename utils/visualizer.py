import numpy as np
import trimesh
import torch
import sys
import visdom

class Visualizer:
    def __init__(self, env, port = 8888):
        self._vis = visdom.Visdom(port = port, env = env)

    def draw_2d_points(self, win, verts, color):
        self._vis.scatter(
            X = verts,
            win = win,
            opts = {
                'markersize': 5,
                'markercolor': color
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
