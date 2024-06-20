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

        # initial step
        for idx, node in enumerate(self.graph.ordered_v_nodes()):
            node.initialize(channel_llr[idx])

        # Perform the decoding process according to the specified sequence
        for iteration in range(self.max_iter):
            print(f"Iteration {iteration + 1}")
            start_time = time.time()
            for cnode_id in self.sequence:
                cnode = self.graph.c_nodes[cnode_id]
                cnode.receive_messages()
                # print("[{}] received".format(cnode_id))
                for vnode_id in cnode.get_neighbors():
                    # print("[{}] connected vnode ids: {}".format(cnode_id, cnode.get_neighbors()))
                    vnode = self.graph.v_nodes[vnode_id]
                    vnode.receive_messages(current_cnode_id=cnode_id)
                    vnode.update_llr()

            # for vnode_id in self.graph.v_nodes.values():
            #     vnode_id.receive_messages()
            #     print("[{}] received".format(vnode_id))

            llr = np.array([node.estimate() for node in self.graph.ordered_v_nodes()])
            print(f"LLR after iteration {iteration + 1}: {llr}")
            estimate = np.array([1 if node_llr < 0 else 0 for node_llr in llr])
            syndrome = self.h.dot(estimate) % 2
            end_time = time.time()
            iteration_times.append(end_time - start_time)
            if not syndrome.any():
                break

        return estimate, llr, not syndrome.any(), iteration_times