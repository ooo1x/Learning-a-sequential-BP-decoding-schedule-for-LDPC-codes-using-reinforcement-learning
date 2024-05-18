from algorithm import BeliefPropagation
from graph import TannerGraph
from channel_models import awgn_llr
import numpy as np
from scipy.sparse import coo_matrix

def load_sparse_matrix(filepath):
    indices = np.load(filepath)
    row_indices, col_indices = indices[0], indices[1]
    data = np.ones_like(row_indices)
    num_rows = np.max(row_indices) + 1
    num_cols = np.max(col_indices) + 1
    return coo_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols)).toarray()

H = load_sparse_matrix('k64_n128_bg2_H_sparse.npy')
original_codeword = np.zeros(H.shape[1]).astype(int)
all_c_nodes_indices = np.arange(H.shape[1], H.shape[1] + H.shape[0])
sequence = np.random.permutation(all_c_nodes_indices)


#Define AWGN channel model parameters
np.random.seed(42)
sigma = 0.6

# BPSK modulation: [1, -1, -1, -1, 1, 1, -1]
transmitted_codeword = 1 - 2 * original_codeword

# Received codeword (with some noise added)
received_codeword = transmitted_codeword + sigma * np.random.randn(len(transmitted_codeword))

# Compute LLR for received symbols in AWGN channel
channel_llr = awgn_llr(sigma, received_codeword)

# Create a Tanner graph for the given H matrix and AWGN channel model
tg = TannerGraph.from_biadjacency_matrix(H, channel_model=lambda x: x)

# Perform decoding
bp = BeliefPropagation(tg, H, max_iter=10, sequence=sequence)
estimate, llr, decode_success = bp.decode(channel_llr)
error_positions = np.logical_xor(original_codeword, estimate)
print("Sent codeword:", original_codeword)
print("Received codeword (noisy BPSK symbols):", received_codeword)
print("Decoded estimate (LLRs):", estimate)
print("Error positions (True indicates a corrected error):", error_positions)
print("Decoding successful:", decode_success)