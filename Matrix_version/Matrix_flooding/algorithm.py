import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import time
from numba import jit


@jit
def calculate_tanh_product(messages: np.ndarray) -> float:
    product_tanh = np.prod(np.tanh(messages / 2))
    safe_product_tanh = np.clip(np.array([product_tanh]), -0.999999, 0.999999)[0]
    return 2 * np.arctanh(safe_product_tanh)


class BeliefPropagation:
    def __init__(self, h: coo_matrix, max_iter: int):
        if isinstance(h, np.ndarray):
            self.H = csr_matrix(h)
        else:
            self.H = h
        self.max_iter = max_iter
        # self.sequence = sequence
        self.num_vnodes = self.H.shape[1]
        self.num_cnodes = self.H.shape[0]
    def compute_message(self, llr, indices, variable_index):
        other_indices = np.array([idx for idx in indices if idx != variable_index], dtype=np.int32)
        message = calculate_tanh_product(llr[other_indices])
        # print(f"Message: {message}")
        return message

    def decode(self, channel_llr: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        if len(channel_llr) != self.num_vnodes:
            raise ValueError("incorrect block size")

        # Initial step
        llr = np.array(channel_llr, dtype=float)
        messages = np.zeros((self.num_cnodes, self.num_vnodes))

        # Perform the decoding process
        for iteration in range(self.max_iter):
            # print(f"Iteration {iteration + 1}")
            new_messages = np.zeros_like(messages)
            # Compute all new messages
            for i in range(self.num_cnodes):
                indices = self.H[i].indices
                for j in indices:
                    new_messages[i, j] = self.compute_message(llr, indices, j)

            # Update LLRs
            for i in range(self.num_cnodes):
                indices = self.H[i].indices
                for j in indices:
                    llr[j] += new_messages[i, j] - messages[i, j]
                messages[i] = new_messages[i]

            # print(f"LLR after iteration {iteration + 1}: {llr}")
            estimate = np.array([1 if l < 0 else 0 for l in llr])
            syndrome = self.H.dot(estimate) % 2
            if not syndrome.any():
                break

        return estimate, llr, not syndrome.any()
