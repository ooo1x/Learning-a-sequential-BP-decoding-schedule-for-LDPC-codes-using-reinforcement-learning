import numpy as np
from scipy.sparse import coo_matrix

# H = np.array([
#     [1, 1, 0, 1, 1, 0, 0],
#     [1, 0, 1, 1, 0, 1, 0],
#     [0, 1, 1, 1, 0, 0, 1]
# ])
def load_sparse_matrix(filepath):
    indices = np.load(filepath)
    row_indices, col_indices = indices[0], indices[1]
    data = np.ones_like(row_indices)
    num_rows = np.max(row_indices) + 1
    num_cols = np.max(col_indices) + 1
    return coo_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols)).toarray()

H = load_sparse_matrix('k64_n128_bg2_H_sparse.npy')
def generate_codewords(H, required_count):
    n = H.shape[1]
    valid_codewords = []
    attempt = 0
    while len(valid_codewords) < required_count:
        c = np.random.randint(0, 2, n)
        if np.all(np.dot(H, c) % 2 == 0):
            if not any(np.array_equal(c, v) for v in valid_codewords):
                valid_codewords.append(c)
        attempt += 1
        if attempt > 100000000:
            break
    return valid_codewords

valid_codewords = generate_codewords(H, 20)
np.savetxt('valid_codewords.txt', valid_codewords, fmt='%d', delimiter=' ')