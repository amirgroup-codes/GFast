import numpy as np

def get_qs_from_delta(delta, q, n):
    '''
    Useful for creating plots: gives alphabet from change in qs (done uniformly across qs)
    '''
    qs = np.array([q] * n)
    threshold = (n-1) * (q-1)
    if (threshold < delta):
        raise ValueError("delta is too big")
    while delta > 0:
        for i in range(1, len(qs)):
            if delta == 0:
                break
            if qs[i] > 1:
                qs[i] -= 1
                delta -=1
    return qs

def get_banned_indices_from_qs(qs, q):
    '''
    Given an array qs you can get the banned indices.
    '''
    delta_dict = {}
    for i, val in enumerate(qs):
        delta = q - val
        if not (delta):
            continue
        else:
            delta_dict[i] = sorted([(q - j-1) for j in range(delta)])
    return delta_dict

def get_qs_from_delta_random(delta, q, n):
    '''
    Gives alphabet change in qs NOT done uniformly across qs
    '''
    qs = np.array([q] * n)
    threshold = (n-1) * (q)
    if (threshold <= delta):
        raise ValueError("delta is too big")
    while delta>0:
        random_idx = np.random.randint(0, n)
        if qs[random_idx] > 1:
            qs[random_idx] -= 1
            delta -= 1
    return qs

def get_qs_from_delta_sitewise(delta, q, n):
    qs = np.array([q] * n)
    i = 0
    while delta > 0 and i < n:
        while qs[i] > 2 and delta > 0:
            qs[i] -= 1
            delta -= 1
        i += 1
    
    return qs

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def calculate_samples(qs, num_subsample, b, num_repeat):
    P = (num_repeat * len(qs)) + 1
    sum = 0
    for i in range(num_subsample):
        sum += P * int(np.prod(qs[len(qs) - (i + 1) * b : len(qs) - i * b]))
    return sum
