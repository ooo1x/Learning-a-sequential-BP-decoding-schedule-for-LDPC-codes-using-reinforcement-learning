import numpy as np
from scipy.sparse import coo_matrix

def load_sparse_matrix(filepath):
    indices = np.load(filepath)
    row_indices, col_indices = indices[0], indices[1]
    data = np.ones_like(row_indices)
    num_rows = np.max(row_indices) + 1
    num_cols = np.max(col_indices) + 1
    matrix = coo_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols)).tocsc()
    print(f"Sparse matrix shape: {matrix.shape}")
    return matrix

def read_codewords(filepath):
    with open(filepath, 'r') as file:
        codewords = file.read().splitlines()
    return codewords

def check_codewords(H, codewords):
    results = []
    print(f"Matrix columns: {H.shape[1]}")
    for codeword in codewords:
        codeword_array = np.array(codeword.split(), dtype=int)
        print(f"Codeword length: {len(codeword_array)}")
        if len(codeword_array) != H.shape[1]:
            print(f"Error: Codeword length ({len(codeword_array)}) does not match matrix columns ({H.shape[1]})")
            print(f"Codeword: {codeword}")
            continue
        syndrome = H.dot(codeword_array) % 2
        if np.all(syndrome == 0):
            results.append(True)
        else:
            results.append(False)
    return results


H = load_sparse_matrix('k64_n128_bg2_H_sparse.npy')
codewords = read_codewords('k64_codewords.txt')
results = check_codewords(H, codewords)

for result in results:
    print(result)
