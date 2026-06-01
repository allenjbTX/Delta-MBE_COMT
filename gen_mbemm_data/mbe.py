import itertools

def generate_combinations(n_frag, order):
    combos = []
    for k in range(1, order + 1):
        combos.extend(itertools.combinations(range(n_frag), k))
    return combos

def recursive_delta(energies, order):
    """Compute non-redundant ΔE for all subsets up to order."""
    delta = {}
    for k in range(1, order + 1):
        for combo in itertools.combinations(range(max(itertools.chain(*energies.keys())) + 1), k):
            if combo not in energies:
                continue
            subtotal = sum(delta[sub] for sub in proper_subsets(combo) if sub in delta)
            delta[combo] = energies[combo] - subtotal
    return delta

def recursive_delta_vector(gradients, order):
    """Compute non-redundant Δgradient arrays for all subsets up to order."""
    delta_g = {}
    max_idx = max(idx for combo in gradients for idx in combo)
    for k in range(1, order + 1):
        for combo in itertools.combinations(range(max_idx + 1), k):
            if combo not in gradients:
                continue
            subtotal = sum(delta_g[sub] for sub in proper_subsets(combo) if sub in delta_g)
            delta_g[combo] = gradients[combo] - subtotal
    return delta_g

def proper_subsets(t):
    for k in range(1, len(t)):
        yield from itertools.combinations(t, k)

    