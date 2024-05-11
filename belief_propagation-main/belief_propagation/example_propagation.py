from algorithm import BeliefPropagation
from graph import TannerGraph
from channel_models import bsc_llr
import numpy as np

H = np.array([[1,1,0,1,1,0,0], [1,0,1,1,0,1,0],[0,1,1,1,0,0,1]])

#Use it to construct a Tanner graph. Assume a BSC channel model with probability p=0.1 for bit flip
model = bsc_llr(0.1)
tg = TannerGraph.from_biadjacency_matrix(H, channel_model=model)

#codeword [1,0,0,0,1,1,0] was sent
received_codeword = np.array([1,0,0,0,1,1,1])

# let us try to correct the error
bp = BeliefPropagation(tg, H, max_iter=10)
estimate, llr, decode_success = bp.decode(received_codeword)
error_positions = np.logical_xor(received_codeword, estimate)
print("Decoded estimate:", estimate)
print("Error positions (True indicates a corrected error):", error_positions)
print("Decoding successful:", decode_success)