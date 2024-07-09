import itertools

s = '012'

permutations = list(itertools.permutations(s))

permutation_dict = {''.join(p): idx for idx, p in enumerate(permutations)}

for perm, idx in permutation_dict.items():
    print(f"permutation: {perm}, int: {idx}")
