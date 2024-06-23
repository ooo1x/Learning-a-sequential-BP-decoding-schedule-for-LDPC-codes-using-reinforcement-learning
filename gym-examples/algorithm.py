import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
import time
from numba import jit


@jit
def calculate_tanh_product(messages: np.ndarray) -> float:
    product_tanh = np.prod(np.tanh(messages / 2))
    safe_product_tanh = np.clip(np.array([product_tanh]), -0.999999, 0.999999)[0]
    return 2 * np.arctanh(safe_product_tanh)

class BeliefPropagation:
    def __init__(self, h: coo_matrix, max_iter: int, sequence: list[int]):
        if isinstance(h, np.ndarray):
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
    def decode(self, channel_llr: NDArray, specific_cnodes: list[int] = None) -> tuple[NDArray, NDArray, bool]:
        if len(channel_llr) != self.num_vnodes:
            raise ValueError("Incorrect block size")
        llr = np.array(channel_llr, dtype=float)
        messages = np.zeros((self.num_cnodes, self.num_vnodes))
        specific_cnodes = specific_cnodes if specific_cnodes is not None else list(range(self.num_cnodes))

        for iteration in range(self.max_iter):
            for i in specific_cnodes:  # 只处理特定的校验节点
                indices = self.H[i].indices
                for j in indices:
                    new_message = self.compute_message(llr, indices, j)
                    llr[j] += new_message - messages[i, j]
                    messages[i, j] = new_message

            estimate = np.array([1 if llr < 0 else 0 for llr in llr])
            syndrome = self.H.dot(estimate) % 2
            if not syndrome.any():
                break

        return estimate, llr, not syndrome.any()

