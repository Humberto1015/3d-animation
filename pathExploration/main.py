import argparse
import numpy as np
from pathSolver import PathSolver

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'SMPL', help = 'The name of dataset')
    parser.add_argument('--source', type = int, default = 268)
    parser.add_argument('--target', type = int, default = 1427)
    parser.add_argument('--num_samples', type = int, default = 30, help = 'Number of samples of the path')
    parser.add_argument('--target_path', type = str, default = './ACAP-sequence/', help = 'The location to save path nodes.')
    parser.add_argument('--mode', type = str, default = 'bezier')
    opt = parser.parse_args()

    solver = PathSolver(opt)
    solver.solve(opt.source, opt.target, mode = opt.mode)
    solver.show_path()

    # save ACAP features to files
    for i in range(solver.seq_ACAP.shape[0]):
        path = opt.target_path + str(i)
        np.save(path, solver.seq_ACAP[i])
    print ('[info] Saved the sequence of ACAP features to %s' % opt.target_path)