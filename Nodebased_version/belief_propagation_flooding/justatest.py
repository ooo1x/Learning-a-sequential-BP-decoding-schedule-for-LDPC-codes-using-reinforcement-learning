import numpy as np
from graph import TannerGraph
from algorithm import BeliefPropagation

def channel_model(x):
    return x

channel_llr = np.array([0.5, -0.3, 0.7, -0.5, 0.2, -0.1])

H = [
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1]
]

graph = TannerGraph.from_biadjacency_matrix(H,channel_model)

bp= BeliefPropagation(graph, H, max_iter=5, sequence=[6, 7, 8])


estimate, llr, decode_success, iteration_times = bp.decode(channel_llr)

print(estimate)