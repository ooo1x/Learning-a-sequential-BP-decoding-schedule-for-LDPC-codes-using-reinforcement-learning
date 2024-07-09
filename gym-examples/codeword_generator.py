import numpy as np

def row_rank(mat):
    g = mat.copy()
    k, n = g.shape

    # Do Gauss-Jordan from left to right
    col = 0
    pr = 0
    skipped = k

    while pr < k:
        if col >= n:
            return pr
        skip = False
        if g[pr, col] == 0:
            # search below for a one
            r = pr + 1
            while r < k and g[r, col] == 0:
                r += 1
            if r < k:
                # swap the rows:
                g[(pr, r), :] = g[(r, pr), :]
            else:
                skip = True
        if not skip:
            # eliminate all other rows
            for r in range(k):
                if r != pr and g[r, col] != 0:
                    g[r, :] = g[r, :] ^ g[pr, :]
            pr += 1
        else:
            skipped += 1
        col += 1
    return pr
def h2g(hp):
    h = hp.copy()
    l, n = h.shape
    k = n - l
    for pr in range(l):
        col = n - l + pr
        if h[pr, col] == 0:
            r = pr + 1
            while r < l and h[r, col] == 0:
                r += 1
            if r < l:
                h[(pr, r), :] = h[(r, pr), :]
            else:
                raise ValueError("Right Part is not full rank, transformation would require column swaps!")
        for r in range(l):
            if r != pr and h[r, col] != 0:
                h[r, :] = h[r, :] ^ h[pr, :]
    g = np.hstack([np.eye(k, dtype=np.uint8), h[:, :k].T])
    return g

def generate_random_codewords(G, num_codewords=100):
    k = G.shape[0]
    info_bits = np.random.randint(0, 2, (num_codewords, k))
    codewords = (info_bits @ G) % 2
    return codewords
