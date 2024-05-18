from algorithm import BeliefPropagation
from graph import TannerGraph
from channel_models import awgn_llr
import numpy as np

H = np.array([[1,1,0,1,1,0,0], [1,0,1,1,0,1,0],[0,1,1,1,0,0,1]])

# Define AWGN channel model parameters
np.random.seed(42)
sigma = 0.6

# Original codeword: [1, 0, 0, 0, 1, 1, 0]
original_codeword = np.array([1, 0, 0, 0, 1, 1, 0])
# BPSK modulation: [1, -1, -1, -1, 1, 1, -1]
transmitted_codeword = 1 - 2 * original_codeword

# Received codeword (with some noise added)
received_codeword = transmitted_codeword + sigma * np.random.randn(len(transmitted_codeword))

# Compute LLR for received symbols in AWGN channel
channel_llr = awgn_llr(sigma, received_codeword)

# Create a Tanner graph for the given H matrix and AWGN channel model
tg = TannerGraph.from_biadjacency_matrix(H, channel_model=lambda x: x)

sequence = [9, 8, 7]
# Perform decoding
bp = BeliefPropagation(tg, H, max_iter=10,sequence=sequence)
estimate, llr, decode_success = bp.decode(channel_llr)
error_positions = np.logical_xor(original_codeword, estimate)
print("Sent codeword:", original_codeword)
print("Received codeword (noisy BPSK symbols):", received_codeword)
print("Decoded estimate (LLRs):", estimate)
print("Error positions (True indicates a corrected error):", error_positions)
print("Decoding successful:", decode_success)