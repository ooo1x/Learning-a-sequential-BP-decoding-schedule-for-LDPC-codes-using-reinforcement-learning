import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from algorithm import BeliefPropagation
from graph import TannerGraph
from channel_models import awgn_llr
import time

def simulate_awgn_bpsk_transmission(H, original_codeword, eb_n0_db, max_iter=10, num_trials=100):
    start_time = time.time()
    n = original_codeword.shape[1]
    ber_results = np.zeros(len(original_codeword))
    all_iteration_times = []
    all_c_nodes_indices = np.arange(H.shape[1], H.shape[1] + H.shape[0])
    sequence = np.random.permutation(all_c_nodes_indices)
    print(f"Sequence of c-node indices for message passing: {sequence}")

    # Calculate the rate R
    k = H.shape[1] - H.shape[0]
    R = k / H.shape[1]

    # Convert Eb/N0 from dB to linear SNR
    eb_n0_linear = 10 ** (eb_n0_db / 10)
    snr_linear = eb_n0_linear * R
    print(f"Currently processing Eb/N0: {eb_n0_db} dB")

    for idx, original_codeword in enumerate(original_codeword):
        print(f"Processing codeword {original_codeword}")
        ber_sum = 0
        iteration_times_per_codeword = []
        for _ in range(num_trials):
            print(f"Trial {_ +1}")
            # BPSK modulation
            transmitted_codeword = 1 - 2 * original_codeword
            sigma = np.sqrt(1 / (2 * snr_linear))

            received_codeword = transmitted_codeword + sigma * np.random.randn(n)
            channel_llr = awgn_llr(sigma, received_codeword)
            tg = TannerGraph.from_biadjacency_matrix(H, channel_model=lambda x: x)
            bp = BeliefPropagation(tg, H, max_iter, sequence=sequence)
            estimate, llr, decode_success, iteration_times = bp.decode(channel_llr)
            iteration_times_per_codeword.append(iteration_times)

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
        all_iteration_times.append(iteration_times_per_codeword)
        end_time = time.time()
        total_duration = end_time - start_time

    # Average BER across all codewords
    average_ber = np.mean(ber_results)
    # with open(f"numba_iteration_times_snr_{snr_db}.txt", 'w') as f:
    #     for trial_times in all_iteration_times:
    #         for times in trial_times:
    #             f.write(" ".join(map(str, times)) + "\n")
    #         trial_average = np.mean([np.mean(t) for t in trial_times])
    #         f.write(f"Avg: {trial_average}\n")
    return average_ber,total_duration

# Load the sparse matrix data
def load_sparse_matrix(filepath):
    indices = np.load(filepath)
    row_indices, col_indices = indices[0], indices[1]
    data = np.ones_like(row_indices)
    num_rows = np.max(row_indices) + 1
    num_cols = np.max(col_indices) + 1
    return coo_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols)).toarray()


H_64 = load_sparse_matrix('k64_n128_bg2_H_sparse.npy')
# H_1024 = load_sparse_matrix('k1024_n2048_CCSDS_H_sparse.npy')
# H_hamming = np.array([[1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]])

# Original codeword
original_codeword_64 = np.loadtxt("k64_codewords.txt")
# original_codeword_1024 = np.zeros(H_1024.shape[1]).astype(int)
# original_codeword_Hamming = np.loadtxt("hamming_codewords.txt")

ber_values = []

# Simulation and plotting
fig, ax = plt.subplots(figsize=(10, 6))
for H, codeword, label in [(H_64, original_codeword_64, 'K64')]:
    ber_values = []
    runtime_data = []
    for eb_n0_db in np.arange(2, 4, 0.5):
        ber, runtime = simulate_awgn_bpsk_transmission(H, codeword, eb_n0_db)
        ber_values.append(ber)
        runtime_data.append(runtime)
    ax.semilogy(np.arange(2, 4, 0.5), ber_values, marker='o', label=f'LDPC {label}')
    with open(f"runtime_and_ber_{label}.txt", 'w') as f:
        for snr, ber, runtime in zip(np.arange(2, 5, 1), ber_values, runtime_data):
            f.write(f"Eb/N0: {snr} dB, BER: {ber:.10f}, Runtime: {runtime:.2f} seconds\n")


ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('BER')
ax.set_title('SNR vs BER for LDPC Codes of Different Lengths')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()
plt.show()