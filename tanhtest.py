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


    def decode(self, channel_llr: NDArray) -> tuple[NDArray, NDArray, bool]:
        print(f"channel llr: {channel_llr}")
        if len(channel_llr) != self.num_vnodes:
            raise ValueError("incorrect block size")

        # Initial step
        llr = np.array(channel_llr, dtype=float)
        messages = np.zeros((self.num_cnodes, self.num_vnodes))

        # Perform the decoding process according to the specified sequence
        for iteration in range(self.max_iter):
            # print(f"Iteration {iteration + 1}")
            for i in self.sequence:
                indices = self.H[i].indices
                # print(f"Processing CNode {i} with connected VNodes {indices}")
                for j in indices:
                    new_message = self.compute_message(llr, indices, j)
                    llr[j] += new_message - messages[i, j]
                    print(f"iteration {iteration}, message {new_message}, llr {llr}, j{j}")
                    # print(f"Update LLR {llr}")# Update LLR by adding new and subtracting old message
                    messages[i, j] = new_message  # Store new message
                    print(f"message {messages[i, j]}")

            # print(f"LLR after iteration {iteration + 1}: {llr}")
            estimate = np.array([1 if llr < 0 else 0 for llr in llr])
            syndrome = self.H.dot(estimate) % 2


        return estimate, llr, not syndrome.any()

H = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 1]], dtype=np.uint8)
original_codeword =[1, 0 ,1, 0 ,1, 0 ,1]




channel_llr= [-0.58957432,  1.76435542, -5.51315037 , 2.38436218 ,-6.65667606, 11.08284505,
 -6.38536211]



sequence = [3,0,1,2]
bp = BeliefPropagation(H, max_iter=1, sequence=sequence)
estimate, llr, decode_success = bp.decode(channel_llr)
print(f"estimate: {estimate}, llr: {llr}, decode_success: {decode_success}")