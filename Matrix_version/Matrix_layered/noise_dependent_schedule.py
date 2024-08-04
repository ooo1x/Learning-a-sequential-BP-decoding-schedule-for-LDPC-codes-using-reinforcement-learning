import numpy as np
from algorithm_dynamic_sequence import BeliefPropagation
import time
from scipy.sparse import coo_matrix
from multiprocessing import Pool, cpu_count
from codeword_generator import generate_random_codewords, h2g, row_rank
import itertools
import random

def awgn_llr(sigma, received_codeword):
    return 2 * received_codeword / (sigma ** 2)
def load_sparse_matrix(filepath):
    indices = np.load(filepath)
    row_indices, col_indices = indices[0], indices[1]
    data = np.ones_like(row_indices,dtype=np.uint8)
    num_rows = np.max(row_indices) + 1
    num_cols = np.max(col_indices) + 1
    return coo_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols))

def simulate_awgn_bpsk_transmission(H, codeword, eb_n0_db, sequences):
    n = codeword.shape[0]
    eb_n0_linear = 10 ** (eb_n0_db / 10)
    Eb = 1
    N0 = Eb / eb_n0_linear
    sigma = np.sqrt(N0 / 2)

    transmitted_codeword = 1 - 2 * codeword
    received_codeword = transmitted_codeword + sigma * np.random.randn(n)
    channel_llr = awgn_llr(sigma, received_codeword)
    bp = BeliefPropagation(H, len(sequences), codeword, sequences)
    estimate, llr, decode_success, this_ber = bp.decode(channel_llr)
    return min(this_ber)


def main():
    start_time_all = time.perf_counter()
    H = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1]], dtype=np.uint8)
    rows, cols = np.where(H == 1)
    values = np.ones(len(rows))
    H = coo_matrix((values, (rows, cols)), shape=(4, 7)).toarray().astype(np.uint8)

    G = h2g(H)
    check = np.dot(G, H.T) % 2
    print("Check H*G^T = 0:", np.all(check == 0))
    original_codeword = generate_random_codewords(G) # 100 codewords

    # Define SNR range in dB
    eb_n0_db = 0
    sequences = list(itertools.permutations([0, 1, 2, 3]))
    all_codewords_sequence_ber = {seq: [] for seq in sequences}
    min_ber_list =[]
    #num_trial = 10

    # for codeword in original_codeword:
    #     print("codeword: ", codeword)
    #     sequence_average_ber = simulate_awgn_bpsk_transmission(H, codeword, eb_n0_db, sequences)
    #     for seq, avg_ber in sequence_average_ber.items():
    #         all_codewords_sequence_ber[seq].append(avg_ber)


    # # Calculate average ber for all codewords
    # overall_sequence_ber = {seq: np.mean(ber_list) for seq, ber_list in all_codewords_sequence_ber.items()}
    # # Find the smallest ber
    # #best_sequence = min(overall_sequence_ber, key=overall_sequence_ber.get)
    # best_avg_ber = overall_sequence_ber[best_sequence]
    #
    # print(f"Best overall sequence: {best_sequence}, with average BER: {best_avg_ber}")

    error_counter = 0
    total_bits = 0

    while error_counter < 1000:
        print(error_counter)
        codeword = generate_random_codewords(G)
        min_ber = simulate_awgn_bpsk_transmission(H, codeword, eb_n0_db, sequences)
        errors_in_this_codeword = min_ber * len(codeword)
        error_counter += errors_in_this_codeword
        total_bits += len(codeword)

    ber = error_counter / total_bits
    print(f"Total bit errors: {error_counter}, Total bits transmitted: {total_bits}, BER: {ber}")
    with open("ber_results.txt", "a") as file:
        file.write(
            f"SNR (dB): {eb_n0_db}, Total bit errors: {error_counter}, Total bits transmitted: {total_bits}, BER: {ber}\n")

    end_time_all = time.perf_counter()
    print(f"Total simulation time: {end_time_all - start_time_all} seconds")


if __name__ == "__main__":
    main()
