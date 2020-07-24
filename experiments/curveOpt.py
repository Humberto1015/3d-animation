import bezier
import numpy as np
import matplotlib.pyplot as plt

# curve
nodes = np.random.rand(4, 2)
curve = bezier.Curve.from_nodes(nodes.T)

t_vals = np.linspace(0, 1, 200)
samples = curve.evaluate_multi(t_vals)

# target points
target_nodes = np.random.rand(3, 2)
xs, ys = target_nodes.T
plt.scatter(xs, ys)

xs, ys = samples
plt.scatter(xs, ys)
plt.show()