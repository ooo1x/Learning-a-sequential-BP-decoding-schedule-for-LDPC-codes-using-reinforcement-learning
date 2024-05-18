import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from algorithm import BeliefPropagation
from graph import TannerGraph
from channel_models import awgn_llr

np.random.seed(42)
def simulate_awgn_bpsk_transmission(H, original_codeword, snr_db, max_iter=10):
    # BPSK modulation
    transmitted_codeword = 1 - 2 * original_codeword
    # Calculate noise standard deviation from SNR
    snr_linear = 10 ** (snr_db / 10)
    Eb = 1  # Energy per bit is 1 for BPSK
    N0 = Eb / snr_linear
    sigma = np.sqrt(N0 / 2)

    # Generate received codeword with AWGN
    received_codeword = transmitted_codeword + sigma * np.random.randn(len(transmitted_codeword))

    # Compute LLR for received symbols in AWGN channel
    channel_llr = awgn_llr(sigma, received_codeword)

    # Create a Tanner graph for the given H matrix
    tg = TannerGraph.from_biadjacency_matrix(H, channel_model=lambda x: x)
    bp = BeliefPropagation(tg, H, max_iter=max_iter)

    # Perform decoding
    estimate, llr, decode_success = bp.decode(channel_llr)

    # Calculate BER
    num_errors = np.sum(np.logical_xor(original_codeword, estimate))
    ber = num_errors / len(original_codeword)

    # Print intermediate results
    error_positions = np.logical_xor(original_codeword, estimate)
    print(f"SNR: {snr_db} dB")
    print("Sent codeword:", original_codeword)
    print("Received codeword (noisy BPSK symbols):", received_codeword)
    print("Decoded estimate:", estimate)
    print("Error positions (True indicates a corrected error):", error_positions)
    print("Decoding successful:", decode_success)
    print(f"BER: {ber}\n")

    return ber


# Load the sparse matrix data
def load_sparse_matrix(filepath):
    indices = np.load(filepath)
    row_indices, col_indices = indices[0], indices[1]
    data = np.ones_like(row_indices)
    num_rows = np.max(row_indices) + 1
    num_cols = np.max(col_indices) + 1
    return coo_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols)).toarray()

H_64 = load_sparse_matrix('k64_n128_bg2_H_sparse.npy')
H_1024 = load_sparse_matrix('k1024_n2048_CCSDS_H_sparse.npy')

# Original codeword
original_codeword_64 = np.zeros(H_64.shape[1]).astype(int)
original_codeword_1024 = np.zeros(H_1024.shape[1]).astype(int)

# Define SNR range in dB
snr_db_range = np.arange(-100, 20, 5)
ber_values = []

# Simulation and plotting
fig, ax = plt.subplots(figsize=(10, 6))
for H, codeword, label in [(H_64, original_codeword_64, '64'), (H_1024, original_codeword_1024, '1024')]:
    ber_values = []
    for snr_db in snr_db_range:
        ber = simulate_awgn_bpsk_transmission(H, codeword, snr_db)
        ber_values.append(ber)
    ax.semilogy(snr_db_range, ber_values, marker='o', label=f'LDPC {label}')

ax.set_xlabel('Es/N0 (dB)')
ax.set_ylabel('BER')
ax.set_title('SNR vs BER for LDPC Codes of Different Lengths')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()
plt.show()