from __future__ import annotations
import numpy as np
import itertools
from typing import Any, Callable
from functools import total_ordering
from abc import ABC, abstractmethod
from numba import njit
import csv


@total_ordering
class Node(ABC):
    """Base class VNodes anc CNodes.
    Derived classes are expected to implement an "initialize" and  method a "message" which should return the message to
    be passed on the graph.
    Nodes are ordered and deemed equal according to their ordering_key.
    """
    _uid_generator = itertools.count()

    @staticmethod
    def reset_uid_generator():
        Node._uid_generator = itertools.count()

    def __init__(self, name: str = "", ordering_key: int = None) -> None:
        """
        :param name: name of node
        """
        self.uid = next(Node._uid_generator)
        self.name = name if name else str(self.uid)
        self.ordering_key = ordering_key if ordering_key is not None else str(self.uid)
        self.neighbors: dict[int, Node] = {}  # keys as senders uid
        self.received_messages: dict[int, Any] = {}  # keys as senders uid, values as messages

    def register_neighbor(self, neighbor: Node) -> None:
        self.neighbors[neighbor.uid] = neighbor

    def __str__(self) -> str:
        if self.name:
            return self.name
        else:
            return str(self.uid)

    def get_neighbors(self) -> list[int]:
        return list(self.neighbors.keys())

    def receive_messages(self) -> None:
        for node_id, node in self.neighbors.items():
            self.received_messages[node_id] = node.message(self.uid)
            print(f"Node {self.uid} received message from Node {node_id}: {node.message(self.uid)}")

    @abstractmethod
    def message(self, requester_uid: int) -> Any:
        pass

    @abstractmethod
    def initialize(self):
        pass

    def __hash__(self):
        return self.uid

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.ordering_key == other.ordering_key

    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.ordering_key < other.ordering_key


class CNode(Node):
    def initialize(self):
        self.received_messages = {node_uid: 0 for node_uid in self.neighbors}

    def message(self, requester_uid: int) -> np.float_:
        messages = [self.received_messages[uid] for uid in self.neighbors if uid != requester_uid]
        product_tanh = np.prod(np.tanh(np.array(messages) / 2))
        safe_product_tanh = np.clip(product_tanh, -0.999999, 0.999999)
        # with open('store.txt', 'wb') as f:
        #     np.save(f, np.array(safe_product_tanh))
        # print('safe_product_tanh',safe_product_tanh)
        return 2 * np.arctanh(safe_product_tanh)

# class CNode(Node):
#     def initialize(self):
#         self.received_messages = {node_uid: 0 for node_uid in self.neighbors}
#
#     @staticmethod
#     @njit
#     def calculate_tanh_product(messages: np.ndarray) -> float:
#         product_tanh = np.prod(np.tanh(messages / 2))
#         product_tanh_array = np.array([product_tanh])
#         safe_product_tanh = np.clip(product_tanh_array, -0.999999, 0.999999)
#         return 2 * np.arctanh(safe_product_tanh[0])
#     def message(self, requester_uid: int) -> np.float_:
#         messages = np.array([self.received_messages[uid] for uid in self.neighbors if uid != requester_uid], dtype=np.float64)
#         return self.calculate_tanh_product(messages)



class VNode(Node):
    def __init__(self, channel_model: Callable, ordering_key: int, name: str = ""):
        self.channel_model = channel_model
        self.channel_llr: np.float_ = None
        super().__init__(name, ordering_key)
        self.last_received_messages = {}

    def initialize(self, channel_symbol):
        self.channel_symbol = channel_symbol
        self.channel_llr = self.channel_model(channel_symbol)
        self.received_messages = {}

    def message(self, requester_uid: int) -> np.float_:
        if requester_uid in self.last_received_messages:
            # Subtract last received message from the current LLR for message calculation
            adjusted_llr = self.channel_llr - self.last_received_messages[requester_uid]
        else:
            adjusted_llr = self.channel_llr
        return adjusted_llr + np.sum(
            msg for uid, msg in self.received_messages.items() if uid != requester_uid
        )

    def receive_messages(self, current_cnode_id=None):
        if current_cnode_id is None:
            return
        message = self.neighbors[current_cnode_id].message(self.uid)
        # Subtract the last received message from this CNode (if exists)
        if current_cnode_id in self.last_received_messages:
            net_message = message - self.last_received_messages[current_cnode_id]
        else:
            net_message = message
        self.received_messages[current_cnode_id] = net_message
        self.last_received_messages[current_cnode_id] = message  # Update last received message
        print(f"VNode {self.uid} received message from CNode {current_cnode_id}: {net_message}")
        self.update_llr()
        print(f"VNode {self.uid} updated LLR: {self.channel_llr}")

    def update_llr(self):
        self.channel_llr += sum(self.received_messages.values())
        self.received_messages.clear()

    def estimate(self) -> np.float_:
        return self.channel_llr