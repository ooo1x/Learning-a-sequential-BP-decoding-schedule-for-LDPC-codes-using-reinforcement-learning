import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
import time
from numba import jit


@jit
def calculate_tanh_product(messages: np.ndarray) -> float:
    product_tanh = np.prod(np.tanh(messages / 2))
    safe_product_tanh = np.clip(np.array([product_tanh]), -0.999999, 0.999999)
    return 2 * np.arctanh(safe_product_tanh)

class BeliefPropagation:
    def __init__(self, h: coo_matrix, max_iter: int, sequence: list[int]):
        if isinstance(h, np.ndarray):
            self.H = csr_matrix(h)
        elif isinstance(h, coo_matrix):
            self.H = csr_matrix(h)
        else:
            self.H = h
        self.max_iter = max_iter
        self.sequence = sequence
        self.num_vnodes = self.H.shape[1]
        self.num_cnodes = self.H.shape[0]

    def compute_message(self, llr, indices, variable_index):
        other_indices = np.array([idx for idx in indices if idx != variable_index], dtype=np.int32)
        message = calculate_tanh_product(llr[other_indices])
        # print(f"Message: {message}")
        return message

    #process specific_nodes
    def decode(self, channel_llr: NDArray, specific_cnode: int) -> tuple[NDArray, NDArray, bool]:
        if len(channel_llr) != self.num_vnodes:
            raise ValueError("Incorrect block size")
        llr = np.array(channel_llr, dtype=float)
        messages = np.zeros((self.num_cnodes, self.num_vnodes))
        residuals = np.zeros((self.num_cnodes, self.num_vnodes))

        for iteration in range(self.max_iter):
            indices = self.H[specific_cnode].indices# variable nodes that connected to the specific check nodes
            for j in indices:
                new_message = self.compute_message(llr, indices, j)
                message_diff = new_message - messages[specific_cnode, j]
                residuals[specific_cnode, j] = np.abs(message_diff)
                llr[j] += message_diff
                messages[specific_cnode, j] = new_message


        # estimate = np.array([1 if llr < 0 else 0 for llr in llr])
        # print("estimate", estimate)
        # syndrome = self.H.dot(estimate) % 2

        return llr, residuals

