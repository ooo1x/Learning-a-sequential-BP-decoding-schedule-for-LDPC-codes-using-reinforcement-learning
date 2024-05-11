import numpy as np
from scipy.sparse import coo_matrix, save_npz

indices = np.load('belief_propagation-main/belief_propagation/k1024_n2048_CCSDS_H_sparse.npy')

row_indices = indices[0]
col_indices = indices[1]

data = np.ones_like(row_indices)

num_rows = row_indices.max() + 1
num_cols = col_indices.max() + 1

matrix = coo_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols))

save_npz('sparse_matrix.npz', matrix.tocsr())

from scipy.sparse import load_npz
loaded_matrix = load_npz('sparse_matrix.npz')
print(loaded_matrix)

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix


indices = np.load('belief_propagation-main/belief_propagation/k1024_n2048_CCSDS_H_sparse.npy')
row_indices = indices[0]
col_indices = indices[1]

data = np.ones_like(row_indices)

matrix = coo_matrix((data, (row_indices, col_indices)))

fig, ax = plt.subplots(figsize=(10, 5))
ax.spy(matrix, markersize=1)
ax.set_title('Sparse Matrix Visualization')
ax.set_xlabel(f'Column Index (Total Columns: {num_cols})')
ax.set_ylabel(f'Row Index (Total Rows: {num_rows})')
plt.show()