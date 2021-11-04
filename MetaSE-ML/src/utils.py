import numpy as np

def negative_sampling_by_label(pos_samples, n_samples):
    pos_set = set(pos_samples)
    all_set = set(list(range(n_samples)))
    neg_samples = list(all_set - pos_set)

    rng = np.random.default_rng()
    rng.shuffle(neg_samples)

    if len(pos_samples) > len(neg_samples):
        return neg_samples
    else:
        return neg_samples[:len(pos_samples)] 
