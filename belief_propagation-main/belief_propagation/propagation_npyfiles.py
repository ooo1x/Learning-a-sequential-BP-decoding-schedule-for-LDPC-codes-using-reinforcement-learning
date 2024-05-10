import numpy as np
from scipy.sparse import coo_matrix
from scipy.linalg import null_space
from algorithm import BeliefPropagation
from graph import TannerGraph
from channel_models import bsc_llr

# Load the sparse matrix data
indices = np.load('k64_n128_bg2_H_sparse.npy')
row_indices, col_indices = indices[0], indices[1]
data = np.ones_like(row_indices)
num_rows = np.max(row_indices) + 1
num_cols = np.max(col_indices) + 1
H = coo_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols)).toarray()

# Construct a Tanner graph
model = bsc_llr(0.1)
tg = TannerGraph.from_biadjacency_matrix(H, channel_model=model)

# Assume some codeword was sent
# Generate a valid codeword by selecting the null space of H if possible
sent_codeword = np.zeros(num_cols, dtype=int)

# Introduce errors
np.random.seed(42)  # For reproducibility
error_vector = np.random.rand(num_cols) < 0.1
received_codeword = (sent_codeword + error_vector) % 2

# Initialize the belief propagation decoder
bp = BeliefPropagation(tg, H, max_iter=2000)
estimate, llr, decode_success = bp.decode(received_codeword)
error_positions = np.logical_xor(received_codeword, estimate)

# Output results
print("Sent codeword:", sent_codeword)
print("Received codeword:", received_codeword)
print("Decoded estimate:", estimate)
print("Error positions (True indicates a corrected error):", error_positions)
print("Decoding successful:", decode_success)