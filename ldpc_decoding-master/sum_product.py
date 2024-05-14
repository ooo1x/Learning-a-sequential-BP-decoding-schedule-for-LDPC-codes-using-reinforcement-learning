import numpy as np
from numba import jit

@jit(nopython=True)
def clip(a, a_min, a_max):
    if a < a_min:
        return a_min
    elif a > a_max:
        return a_max
    else:
        return a

@jit(nopython=True)
def update_check_nodes(H, M, B, B_size):
    n_c, n_v = H.shape
    E = np.zeros((n_c, n_v))

    for j in range(n_c):
        for idx in range(B_size[j]):
            i = B[j, idx]
            a = 1.0
            for kdx in range(B_size[j]):
                k = B[j, kdx]
                if k != i:
                    a *= np.tanh(M[j, k] / 2)
            a = clip(a, -0.999999, 0.999999)
            E[j, i] = 2 * np.arctanh(a)

    return E

def update_variable_nodes(H, E, R, n_c, n_v):
    L = np.zeros(n_v)

    for i in range(n_v):
        temp = 0
        for j in range(n_c):
            temp += E[j, i]
        L[i] = R[i] + temp

    return L

def ldpc_soft_decision(H, code_0, code_1, max_iter):
    # Initialization
    R = code_0  # LLR for bit 0
    R0 = code_1  # LLR for bit 1
    n_c, n_v = H.shape  # Number of rows (check nodes) and columns (variable nodes) in H
    M = np.zeros((n_c, n_v))  # Initialize message matrix M with zeros

    # Build list B: each element is a list of variable nodes connected to a check node
    B = -np.ones((n_c, n_v), dtype=np.int32)
    B_size = np.zeros(n_c, dtype=np.int32)
    print("H matrix:", H)

    for j in range(n_c):
        idx = 0
        for i in range(n_v):
            if H[j, i] == 1:
                B[j, idx] = i
                idx += 1
        B_size[j] = idx
    print("B matrix :", B)
    print("B_size :", B_size)

    # Initialize message matrix M
    for i in range(n_v):
        for j in range(n_c):
            if H[j, i] == 1:
                M[j, i] = R0[i]
    print("M matrix after initialization:", M)

    # Iterative process
    for r in range(max_iter):
        print(f"Iteration {r + 1}")
        # Horizontal step: update check nodes row by row
        E = np.zeros((n_c, n_v))
        for j in range(n_c):
            E_j = update_check_nodes(H[j:j+1, :], M[j:j+1, :], B[j:j+1, :], B_size[j:j+1])
            E[j:j+1, :] = E_j
        # Vertical step: update variable nodes
        L = update_variable_nodes(H, E, R, n_c, n_v)

        # Bit message update: update matrix M
        for i in range(n_v):
            for j in range(n_c):
                if H[j, i] != 0:
                    M[j, i] = R[i] + sum(E[:, i]) - E[j, i]
        print("M matrix after initialization:", M)

        # Stopping criterion based on syndrome check
        c = (L <= 0).astype(int)
        syndrome = H.dot(c) % 2
        print(f"Syndrome after iteration {r + 1}: {syndrome}")
        if not syndrome.any():
            return L  # Decoding successful, return LLRs

    return L  # Return final LLRs
