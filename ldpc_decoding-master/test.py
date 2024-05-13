import numpy as np
from scipy.sparse import coo_matrix
import sum_product as sp

# Load the sparse matrix data
indices = np.load('k64_n128_bg2_H_sparse.npy')
row_indices, col_indices = indices[0], indices[1]
data = np.ones_like(row_indices)
num_rows = np.max(row_indices) + 1
num_cols = np.max(col_indices) + 1
H = coo_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols)).toarray()

# Construct a Tanner graph
sent_codeword = np.zeros((num_cols)).astype(int)
result = sent_codeword.dot(H.T) % 2
if np.all(result == 0):
    print("Valid sent codeword.")

# Simulate received codeword with noise (using BPSK modulation and AWGN noise)
np.random.seed(42)  # For reproducibility

# Convert sent codeword to BPSK symbols
z = 2 * sent_codeword - 1  # BPSK modulation: 0 -> -1, 1 -> 1

# Simulate channel with AWGN noise
sigma = 1  # Noise standard deviation
noisy_received = z + sigma * np.random.randn(num_cols)  # Add AWGN noise

# Calculate LLRs for received bits
f1 = 1 / (1 + np.exp(-2 * noisy_received / sigma**2))  # LLR for bit being 1
f0 = 1 - f1  # LLR for bit being 0

# Initialize the belief propagation decoder and decode
max_iter = 10
estimate = sp.ldpc_soft_decision(H, f0, f1, max_iter)

# Convert LLRs to hard decisions (0 or 1)
decoded_codeword = (estimate < 0).astype(int)  # LLR > 0 means bit is 1, LLR <= 0 means bit is 0

# Check for decoding success
error_positions = sent_codeword != decoded_codeword
decode_success = np.all((decoded_codeword.dot(H.T) % 2) == 0)

# Output results
print("Sent codeword:", sent_codeword)
print("Received codeword (noisy BPSK symbols):", noisy_received)
print("Decoded estimate (LLRs):", estimate)
print("Decoded codeword:", decoded_codeword)
print("Error positions (True indicates a corrected error):", error_positions)
print("Decoding successful:", decode_success)
