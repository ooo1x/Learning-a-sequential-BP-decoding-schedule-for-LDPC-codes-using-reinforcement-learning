import numpy as np
from algorithm import BeliefPropagation
import time
from scipy.sparse import coo_matrix
from multiprocessing import Pool, cpu_count
from codeword_generator import generate_random_codewords, h2g, row_rank
import itertools

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
    H, original_codeword, eb_n0_db, max_iter, sequence = args
    n = len(original_codeword)
    total_errors = 0
    total_bits = 0

    while total_errors < 1000:
        eb_n0_linear = 10 ** (eb_n0_db / 10)
        transmitted_codeword = 1 - 2 * original_codeword
        Eb = 1
        N0 = Eb / eb_n0_linear
        sigma = np.sqrt(N0 / 2)

        received_codeword = transmitted_codeword + sigma * np.random.randn(n)
        channel_llr = awgn_llr(sigma, received_codeword)
        bp = BeliefPropagation(H, max_iter, sequence=sequence)
        estimate, llr, decode_success = bp.decode(channel_llr)

        num_errors = np.sum(np.logical_xor(original_codeword, estimate))
        total_errors += num_errors
        total_bits += n

    ber = total_errors / total_bits if total_bits > 0 else 0
    return eb_n0_db, total_errors, total_bits, ber



def main():
    start_time_all = time.perf_counter()
    H = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1]], dtype=np.uint8)
    rows, cols = np.where(H == 1)
    values = np.ones(len(rows))
    H = coo_matrix((values, (rows, cols)), shape=(4, 7)).toarray().astype(np.uint8)
    G = h2g(H)
    original_codeword = generate_random_codewords(G)

    snr_values = [0, 0.5, 1, 1.5, 2]
    max_iter = 1
    sequence = [3,2,1,0]

    pool = Pool(processes=cpu_count())
    results = pool.map(simulate_awgn_bpsk_transmission, [(H, original_codeword, snr, max_iter, sequence) for snr in snr_values])
    pool.close()
    pool.join()


    for result in results:
        eb_n0_db, total_errors, total_bits, ber = result
        print(f"SNR: {eb_n0_db} dB, BER: {ber}\n")


    end_time_all = time.perf_counter()
    total_duration_program = end_time_all - start_time_all
    print(f"Total runtime of the program: {total_duration_program:.2f} seconds")

if __name__ == "__main__":
    main()