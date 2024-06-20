import numpy as np
from algorithm import BeliefPropagation
import time
from scipy.sparse import coo_matrix
from multiprocessing import Pool, cpu_count
from codeword_generator import generate_random_codewords, h2g, row_rank

def awgn_llr(sigma, received_codeword):
    return 2 * received_codeword / (sigma ** 2)
def load_sparse_matrix(filepath):
    indices = np.load(filepath)
    row_indices, col_indices = indices[0], indices[1]
    data = np.ones_like(row_indices,dtype=np.uint8)
    num_rows = np.max(row_indices) + 1
    num_cols = np.max(col_indices) + 1
    return coo_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols))

def simulate_awgn_bpsk_transmission(args):
    H, original_codeword, eb_n0_db, max_iter, num_trials = args
    start_time = time.perf_counter()
    n = original_codeword.shape[1]
    ber_results = np.zeros(len(original_codeword))
    bler_results = np.zeros(len(original_codeword))
    all_c_nodes_indices = np.arange(H.shape[0])
    # sequence = all_c_nodes_indices
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
            bp = BeliefPropagation(H, max_iter)
            estimate, llr, decode_success = bp.decode(channel_llr)

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
    H = H.toarray().astype(np.uint8)
    # H = np.array([[1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]])
    # rows, cols = np.where(H == 1)
    # values = np.ones(len(rows))
    # H = coo_matrix((values, (rows, cols)), shape=(3, 7))
    # Codewords
    # original_codeword = np.loadtxt("hamming_codewords.txt")*0
    # original_codeword = np.loadtxt("k64_codewords.txt") * 0
    G = h2g(H)
    check = np.dot(G, H.T) % 2
    print("Check H*G^T = 0:", np.all(check == 0))
    # rank_G = row_rank(G)
    # rank_H = row_rank(H)
    # print("Rank of G:", rank_G)
    # print("Rank of H:", rank_H)
    original_codeword = generate_random_codewords(G)
    original_codeword = original_codeword[np.random.choice(original_codeword.shape[0], 10, replace=False), :]
    # print("selected_codewords:", selected_codewords)

    # Define SNR range in dB
    eb_n0_db_range = np.arange(0, 3, 0.5)
    max_iter = 10
    num_trials = 1000

    args = [(H, original_codeword, db, max_iter, num_trials) for db in eb_n0_db_range]
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(simulate_awgn_bpsk_transmission, args)

    ber_values = []
    runtime_data = []
    bler_values = []

    with open("../Matrix_layered/ber_snr_results_flooding_with_multi.txt", 'w') as f:
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
