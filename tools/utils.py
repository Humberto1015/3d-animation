import numpy as np
import torch
import sys
import visdom
import argparse


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
