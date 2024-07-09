import numpy as np
from algorithm import BeliefPropagation
from scipy.sparse import coo_matrix

def channel_model(x):
    return x

channel_llr = np.array([0.5, -0.3, 0.7, -0.5, 0.2, -0.1])

H_data = np.array([
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1]
])
rows, cols = np.where(H_data == 1)
values = np.ones(len(rows))
H = coo_matrix((values, (rows, cols)), shape=(3, 6))


bp= BeliefPropagation(H, max_iter=5)

estimate, llr, decode_success= bp.decode(channel_llr)

print(estimate)



