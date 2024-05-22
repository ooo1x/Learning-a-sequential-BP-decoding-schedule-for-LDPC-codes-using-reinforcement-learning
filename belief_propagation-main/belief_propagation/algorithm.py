from graph import TannerGraph
from numpy.typing import ArrayLike, NDArray
import numpy as np
import time

class BeliefPropagation:
    def __init__(self, graph: TannerGraph, h: ArrayLike, max_iter: int, sequence: list[int]):
        self.h = np.array(h)
        self.graph = graph
        self.n = len(graph.v_nodes)
        self.max_iter = max_iter
        self.sequence = sequence

    def decode(self, channel_llr) -> tuple[NDArray, NDArray, bool]:
        if len(channel_llr) != self.n:
            raise ValueError("incorrect block size")
        iteration_times = []

        # Initialize variable nodes with channel LLR
        for idx, node in enumerate(self.graph.ordered_v_nodes()):
            node.initialize(channel_llr[idx])

        # Send initial channel-based messages to check nodes
        for node in self.graph.c_nodes.values():
            node.receive_messages()

        # Perform the decoding process according to the specified sequence
        for iteration in range(self.max_iter):
            print(f"Iteration {iteration + 1}")
            start_time = time.time()

            # Update check nodes in the specified sequence and update variable nodes immediately after each check node
            for cnode_id in self.sequence:
                self.graph.c_nodes[cnode_id].receive_messages()
                for vnode in self.graph.v_nodes.values():
                    vnode.receive_messages()
                    vnode.update_llr()

            llr = np.array([node.estimate() for node in self.graph.ordered_v_nodes()])
            # print(f"LLR after iteration {iteration + 1}: {llr}")
            estimate = np.array([1 if node_llr < 0 else 0 for node_llr in llr])
            # print(f"Estimate after iteration {iteration + 1}: {estimate}")
            syndrome = self.h.dot(estimate) % 2
            # print(f"Syndrome after iteration {iteration + 1}: {syndrome}")
            end_time = time.time()
            iteration_times.append(end_time - start_time)

            if not syndrome.any():
                return estimate, llr, True, iteration_times

        return estimate, llr, False, iteration_times

