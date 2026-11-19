import os

def build_slater_koster_dict(directory, elems):
    sldict = {}
    for elem1 in elems:
        for elem2 in elems:
            key = elem1 + "-" + elem2
            if key not in sldict:
                filename = f"{elem1}-{elem2}.skf"
                filepath = os.path.join(directory, filename)
                sldict[key] = filepath
    return sldict

def build_hubbard_derivs_dict(elems):
    base_dict = {'O':-0.1575, 'H':-0.1857, 'N':-0.1535, 'C':-0.1492, 'S':-0.11, 'Mg':-0.02}
    return {elem: base_dict.get(elem, 0) for elem in elems}

