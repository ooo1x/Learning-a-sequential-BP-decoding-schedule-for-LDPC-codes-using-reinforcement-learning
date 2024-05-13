import numpy as np
from numba import jit

def update_check_nodes(H, M, B, j):
    n_c, n_v = H.shape
    E = np.zeros((n_c, n_v))

    for i in range(n_v):
        if H[j, i] != 0:
            a = 1.0
            for k in B[j]:
                if k != i:
                    a *= np.tanh(M[j, k] / 2)
            a = np.clip(a, -0.999999, 0.999999)
            E[j, i] = 2 * np.arctanh(a)

    return E

def update_variable_nodes(H, E, R, n_c, n_v):
    L = np.zeros(n_v)
    c = np.zeros(n_v, int)

    for i in range(n_v):
        temp = 0
        for j in range(n_c):
            temp += E[j, i]
        L[i] = R[i] + temp
        c[i] = 1 if L[i] <= 0 else 0

    return L, c


def ldpc_soft_decision(H, code_0, code_1, max_iter):
    # Initialization
    R = code_0  # LLR of bit 0
    R0 = code_1  # LLR of bit 1
    n_c, n_v = H.shape  # number of rows (check nodes) and columns (variable nodes) of H
    M = np.zeros((n_c, n_v))  # Initialize message matrix M with zeros
    B = []  # Initialize list B to store variable nodes connected to each check node

    # Initialize the message matrix M
    for i in range(n_v):
        for j in range(n_c):
            if H[j, i] == 1:
                M[j, i] = R0[i]

    # Build list B: each element is a list of variable nodes connected to a check node
    for j in range(n_c):
        temp = []
        for i in range(n_v):
            if H[j, i] == 1:
                temp.append(i)
        B.append(temp)

    # Iterative process
    for r in range(max_iter):
        # Horizontal step: update check nodes one by one
        E = np.zeros((n_c, n_v))
        for j in range(n_c):
            E[j, :] = update_check_nodes(H, M, B, j)[j, :]

        # Vertical step: update variable nodes and make hard decisions in parallel
        L, c = update_variable_nodes(H, E, R, n_c, n_v)

        # Stop criteria: check if hard decisions satisfy all check equations
        if np.all(np.dot(c, H.T) % 2 == 0):
            return L  # Decoding successful, return LLRs

        # Bit message update: update matrix M
        for j in range(n_c):
            for i in range(n_v):
                if H[j, i] != 0:
                    tanh_sum = np.prod(np.tanh(E[j, B[j]] / 2)) / np.tanh(E[j, i] / 2)
                    tanh_sum = np.clip(tanh_sum, -0.999999, 0.999999)
                    M[j, i] = 2 * np.arctanh(tanh_sum) + R[i]

    return L  # Return final LLRs
