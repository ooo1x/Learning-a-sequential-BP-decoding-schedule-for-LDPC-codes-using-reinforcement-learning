import numpy as np
import sum_product as sp

# LDPC matrix
H = np.array([[1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]])

# Original sent codeword [1,0,0,0,1,1,0]
sent_codeword = np.array([1, 0, 0, 0, 1, 1, 0])
sigma = 0.3

# Convert sent codeword to BPSK symbols
z = 2 * sent_codeword - 1  # BPSK modulation: 0 -> -1, 1 -> 1

# Simulate channel with AWGN noise
np.random.seed(42)  # For reproducibility
noisy_received = z + sigma * np.random.randn(len(sent_codeword))  # Add AWGN noise

# Calculate LLRs for received bits
llr = 2 * noisy_received / sigma**2

# Initialize the belief propagation decoder and decode
max_iter = 10
estimate = sp.ldpc_soft_decision(H, -llr, llr, max_iter)

# Convert LLRs to hard decisions (0 or 1)
decoded_codeword = (estimate < 0).astype(int)  # LLR > 0 means bit is 0, LLR <= 0 means bit is 1

# Check for decoding success
decode_success = np.all((decoded_codeword.dot(H.T) % 2) == 0)

# Output results
print("Sent codeword:", sent_codeword)
print("Received codeword (noisy BPSK symbols):", noisy_received)
print("Decoded estimate (LLRs):", estimate)
print("Decoded codeword:", decoded_codeword)
print("Decoding successful:", decode_success)
