import numpy as np
import matplotlib.pyplot as plt
from algorithm import BeliefPropagation
from graph import TannerGraph
from channel_models import awgn_llr

def simulate_awgn_bpsk_transmission(H, original_codeword, snr_db, max_iter=10):
    # BPSK modulation
    transmitted_codeword = 1 - 2 * original_codeword
    # Calculate noise standard deviation from SNR
    snr_linear = 10 ** (snr_db / 10)
    Eb = 1  # Energy per bit is 1 for BPSK
    N0 = Eb / snr_linear
    sigma = np.sqrt(N0 / 2)

    # Received codeword (with some noise added)
    received_codeword = transmitted_codeword + sigma * np.random.randn(len(transmitted_codeword))

    # Compute LLR for received symbols in AWGN channel
    channel_llr = awgn_llr(sigma, received_codeword)

    # Create a Tanner graph for the given H matrix and AWGN channel model
    tg = TannerGraph.from_biadjacency_matrix(H, channel_model=lambda x: x)

    # Perform decoding
    bp = BeliefPropagation(tg, H, max_iter)
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

# Define the H matrix and original codeword
H = np.array([[1,1,0,1,1,0,0], [1,0,1,1,0,1,0],[0,1,1,1,0,0,1]])
original_codeword = np.array([1, 0, 0, 0, 1, 1, 0])

# Define SNR range in dB
snr_db_range = np.arange(-50, 10, 10)
ber_values = []

# Simulation and plotting
fig, ax = plt.subplots(figsize=(10, 6))
ber_values = []
for snr_db in snr_db_range:
    ber = simulate_awgn_bpsk_transmission(H, original_codeword, snr_db)
    ber_values.append(ber)
ax.semilogy(snr_db_range, ber_values, marker='o', label='LDPC Code')

ax.set_xlabel('SNR(Eb/N0) (dB)')
ax.set_ylabel('BER')
ax.set_ylim(0.00001, 1)
ax.set_title('SNR vs BER for LDPC Code')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()
plt.show()
