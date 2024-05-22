import numpy as np
import matplotlib.pyplot as plt
from algorithm import BeliefPropagation
from graph import TannerGraph
from channel_models import awgn_llr


def simulate_awgn_bpsk_transmission(H, original_codeword, snr_db, max_iter=10, num_trials=100):
    n = original_codeword.shape[1]
    ber_results = np.zeros(len(original_codeword))

    for idx, original_codeword in enumerate(original_codeword):
        print(f"Processing codeword {original_codeword}")
        ber_sum = 0
        for _ in range(num_trials):
            print(f"Trial {_ +1}")
            # BPSK modulation
            transmitted_codeword = 1 - 2 * original_codeword
            snr_linear = 10 ** (snr_db / 10)
            Eb = 1
            N0 = Eb / snr_linear
            sigma = np.sqrt(N0 / 2)

            received_codeword = transmitted_codeword + sigma * np.random.randn(n)
            channel_llr = awgn_llr(sigma, received_codeword)
            tg = TannerGraph.from_biadjacency_matrix(H, channel_model=lambda x: x)
            sequence = [7, 8, 9]
            bp = BeliefPropagation(tg, H, max_iter, sequence=sequence)
            estimate, llr, decode_success = bp.decode(channel_llr)

            # Calculate BER for this trial
            num_errors = np.sum(np.logical_xor(original_codeword, estimate))
            ber = num_errors / n
            ber_sum += ber
            # # Print intermediate results
            # error_positions = np.logical_xor(original_codeword, estimate)
            # print(f"SNR: {snr_db} dB")
            # print("Sent codeword:", original_codeword)
            # print("Received codeword (noisy BPSK symbols):", received_codeword)
            # print("Decoded estimate:", estimate)
            # print("Error positions (True indicates a corrected error):", error_positions)
            print("Decoding successful:", decode_success)

        # Average BER over all trials for this codeword
        ber_results[idx] = ber_sum / num_trials
    # Average BER across all codewords
    average_ber = np.mean(ber_results)
    return average_ber

def plot_iteration_times(iteration_times):
    plt.figure(figsize=(10, 5))
    plt.plot(iteration_times, marker='o', linestyle='-', color='b')
    plt.title("Time per Iteration")
    plt.xlabel("Iteration Number")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.show()

# Define the H matrix
H = np.array([[1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]])

# Codewords
original_codeword = np.loadtxt("valid_codewords.txt")

# Define SNR range in dB
snr_db_range = np.arange(-2.5, 2.5, 0.5)
ber_values = []

# Simulation and plotting
fig, ax = plt.subplots(figsize=(10, 6))
for snr_db in snr_db_range:
    average_ber = simulate_awgn_bpsk_transmission(H, original_codeword, snr_db)
    ber_values.append(average_ber)

ax.semilogy(snr_db_range, ber_values, marker='o', label='H')
ax.set_xlabel('SNR(Eb/N0) (dB)')
ax.set_ylabel('BER')
ax.set_ylim(0.001, 1)
ax.set_title('SNR vs BER for LDPC Code with Multiple Codewords')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()
plt.show()
