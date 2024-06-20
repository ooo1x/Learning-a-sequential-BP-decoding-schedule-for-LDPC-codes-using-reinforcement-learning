# import numpy as np
# import matplotlib.pyplot as plt
# from algorithm import BeliefPropagation
# from graph import TannerGraph
# from channel_models import awgn_llr
# import time
# from scipy.sparse import coo_matrix
# import multiprocessing
#
# def load_sparse_matrix(filepath):
#     indices = np.load(filepath)
#     row_indices, col_indices = indices[0], indices[1]
#     data = np.ones_like(row_indices)
#     num_rows = np.max(row_indices) + 1
#     num_cols = np.max(col_indices) + 1
#     return coo_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols)).toarray()
#
# def simulate_awgn_bpsk_transmission(H, original_codeword, eb_n0_db, max_iter=10, num_trials=1000):
#     start_time = time.time()
#     n = original_codeword.shape[1]
#     ber_results = np.zeros(len(original_codeword))
#     bler_results = np.zeros(len(original_codeword))
#     all_iteration_times = []
#     all_c_nodes_indices = np.arange(H.shape[1], H.shape[1] + H.shape[0])
#     sequence = all_c_nodes_indices
#     # print(f"Sequence of c-node indices for message passing: {sequence}")
#
#     eb_n0_linear = 10 ** (eb_n0_db / 10)
#     print(f"Currently processing Eb/N0: {eb_n0_db} dB")
#
#     for idx, original_codeword in enumerate(original_codeword):
#         # print(f"Processing codeword {original_codeword}")
#         ber_sum = 0
#         block_errors = 0
#         iteration_times_per_codeword = []
#         for _ in range(num_trials):
#             # print(f"Trial {_ +1}")
#             # BPSK modulation
#             transmitted_codeword = 1 - 2 * original_codeword
#             Eb = 1
#             N0 = Eb / eb_n0_linear
#             sigma = np.sqrt(N0 / 2)
#
#             received_codeword = transmitted_codeword + sigma * np.random.randn(n)
#             channel_llr = awgn_llr(sigma, received_codeword)
#             # channel_llr_hard = -0.5*(np.sign(channel_llr)-1)
#             tg = TannerGraph.from_biadjacency_matrix(H, channel_model=lambda x: x)
#             bp = BeliefPropagation(tg, H, max_iter, sequence=sequence)
#             estimate, llr, decode_success, iteration_times = bp.decode(channel_llr)
#
#             # Calculate BER for this trial
#             num_errors = np.sum(np.logical_xor(original_codeword, estimate))
#             ber = num_errors / n
#             ber_sum += ber
#             if not decode_success:
#                 block_errors += 1
#
#         # Average BER over all trials for this codeword
#         ber_results[idx] = ber_sum / num_trials
#         bler_results[idx] = block_errors / num_trials
#         all_iteration_times.append(iteration_times_per_codeword)
#
#     end_time = time.time()
#     total_duration = end_time - start_time
#
#     # Average BER across all codewords
#     average_ber = np.mean(ber_results)
#     average_bler = np.mean(bler_results)
#     return average_ber,total_duration, average_bler
#
# start_time_all = time.time()
# H = load_sparse_matrix('k64_n128_bg2_H_sparse.npy')
# # H = np.array([[1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]])
# # Codewords
# # original_codeword = np.loadtxt("hamming_codewords.txt")*0
# original_codeword = np.loadtxt("k64_codewords.txt")*0
#
# # Define SNR range in dB
# eb_n0_db_range = np.arange(0, 3, 1)
# ber_values = []
# runtime_data = []
#
# # Simulation and plotting
# with open("k64_ber_snr_results_layered_with_numba.txt", 'w') as f:
#     for eb_n0_db in eb_n0_db_range:
#         average_ber, runtime , average_bler = simulate_awgn_bpsk_transmission(H, original_codeword, eb_n0_db)
#         ber_values.append(average_ber)
#         runtime_data.append(runtime)
#         f.write(f"{eb_n0_db} {average_ber} {average_bler} {runtime:.2f}\n")
#
# end_time_all = time.time()
# total_duration_all = end_time_all - start_time_all
# print(f"Total runtime of the program: {total_duration_all:.2f} seconds")

import numpy as np
import matplotlib.pyplot as plt
from algorithm import BeliefPropagation
from graph import TannerGraph
from channel_models import awgn_llr
import time
from scipy.sparse import coo_matrix
from multiprocessing import Pool, cpu_count

def load_sparse_matrix(filepath):
    indices = np.load(filepath)
    row_indices, col_indices = indices[0], indices[1]
    data = np.ones_like(row_indices)
    num_rows = np.max(row_indices) + 1
    num_cols = np.max(col_indices) + 1
    return coo_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols)).toarray()

def simulate_awgn_bpsk_transmission(args):
    H, original_codeword, eb_n0_db, max_iter, num_trials = args
    start_time = time.perf_counter()
    n = original_codeword.shape[1]
    ber_results = np.zeros(len(original_codeword))
    bler_results = np.zeros(len(original_codeword))
    all_c_nodes_indices = np.arange(H.shape[1], H.shape[1] + H.shape[0])
    sequence = all_c_nodes_indices
    # print(f"Sequence of c-node indices for message passing: {sequence}")

    eb_n0_linear = 10 ** (eb_n0_db / 10)
    print(f"Currently processing Eb/N0: {eb_n0_db} dB")

    for idx, original_codeword in enumerate(original_codeword):
        # print(f"Processing codeword {original_codeword}")
        ber_sum = 0
        block_errors = 0
        iteration_times_per_codeword = []
        for _ in range(num_trials):
            # print(f"Trial {_ +1}")
            # BPSK modulation
            transmitted_codeword = 1 - 2 * original_codeword
            Eb = 1
            N0 = Eb / eb_n0_linear
            sigma = np.sqrt(N0 / 2)

            received_codeword = transmitted_codeword + sigma * np.random.randn(n)
            channel_llr = awgn_llr(sigma, received_codeword)
            # channel_llr_hard = -0.5*(np.sign(channel_llr)-1)
            tg = TannerGraph.from_biadjacency_matrix(H, channel_model=lambda x: x)
            bp = BeliefPropagation(tg, H, max_iter, sequence=sequence)
            estimate, llr, decode_success, iteration_times = bp.decode(channel_llr)

            # Calculate BER for this trial
            num_errors = np.sum(np.logical_xor(original_codeword, estimate))
            ber = num_errors / n
            ber_sum += ber
            if not decode_success:
                block_errors += 1

        ber_results[idx] = ber_sum / num_trials
        bler_results[idx] = block_errors / num_trials

    end_time = time.perf_counter()
    total_duration = end_time - start_time

    # Average BER across all codewords
    average_ber = np.mean(ber_results)
    average_bler = np.mean(bler_results)
    return eb_n0_db, average_ber, average_bler, total_duration


def main():
    start_time_all = time.perf_counter()
    H = load_sparse_matrix('k64_n128_bg2_H_sparse.npy')
    # H = np.array([[1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]])
    # Codewords
    # original_codeword = np.loadtxt("hamming_codewords.txt")*0
    original_codeword = np.loadtxt("k64_codewords.txt") * 0

    # Define SNR range in dB
    eb_n0_db_range = np.arange(0, 3, 1)
    max_iter = 10
    num_trials = 1

    args = [(H, original_codeword, db, max_iter, num_trials) for db in eb_n0_db_range]
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(simulate_awgn_bpsk_transmission, args)

    ber_values = []
    runtime_data = []
    bler_values = []

    with open("k64_ber_snr_results_layered_with_numba_multi_1.txt", 'w') as f:
        for result in results:
            eb_n0_db, average_ber, average_bler, total_duration = result
            ber_values.append(average_ber)
            bler_values.append(average_bler)
            runtime_data.append(total_duration)
            f.write(f"{eb_n0_db} {average_ber} {average_bler} {total_duration:.2f}\n")

    end_time_all = time.perf_counter()
    total_duration_program = end_time_all - start_time_all
    print(f"Total runtime of the program: {total_duration_program:.2f} seconds")

if __name__ == "__main__":
    main()
