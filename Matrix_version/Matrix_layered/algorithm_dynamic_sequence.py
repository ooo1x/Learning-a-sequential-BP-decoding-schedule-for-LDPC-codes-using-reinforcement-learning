import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
import time
from numba import jit
import itertools
@jit
def calculate_tanh_product(messages: np.ndarray) -> float:
    product_tanh = np.prod(np.tanh(messages / 2))
    safe_product_tanh = np.clip(np.array([product_tanh]), -0.999999, 0.999999)[0]
    return 2 * np.arctanh(safe_product_tanh)

class BeliefPropagation:
    def __init__(self, h, max_iter, original_codeword,sequences):
        if isinstance(h, np.ndarray):
            self.H = csr_matrix(h)
        else:
            self.H = h
        self.max_iter = max_iter
        self.num_vnodes = self.H.shape[1]
        self.num_cnodes = self.H.shape[0]
        self.original_codeword = original_codeword
        self.sequences = sequences
        self.sequence_index = 0


    def compute_message(self, llr, indices, variable_index):
        other_indices = np.array([idx for idx in indices if idx != variable_index], dtype=np.int32)
        message = calculate_tanh_product(llr[other_indices])
        # print(f"Message: {message}")
        return message

    def decode(self, channel_llr):
        if len(channel_llr) != self.num_vnodes:
            raise ValueError("incorrect block size")

        # Initial step
        llr = np.array(channel_llr, dtype=float)
        messages = np.zeros((self.num_cnodes, self.num_vnodes))
        ber_per_iteration = []

        # Perform the decoding process according to the specified sequence

        for _ in range(len(self.sequences)):
            sequence = self.sequences[self.sequence_index]
            #print("processing", sequence)
            for i in sequence:
                indices = self.H[i].indices
                for j in indices:
                    new_message = self.compute_message(llr, indices, j)
                    llr[j] += new_message - messages[i, j]
                    messages[i, j] = new_message
            self.sequence_index = (self.sequence_index + 1) % len(self.sequences)

            estimate = np.array([1 if x < 0 else 0 for x in llr])
            num_errors = np.sum(np.logical_xor(self.original_codeword, estimate))
            ber = num_errors / self.num_vnodes
            ber_per_iteration.append(ber)
            #print("ber per iteration", ber_per_iteration)
            syndrome = self.H.dot(estimate) % 2

        return estimate, llr, not syndrome.any(), ber_per_iteration



