'''
Methods for the reconstruction engine; specifically, to:

1. carry out singleton detection
2. get the cardinalities of all bins in a subsampling group (debugging only).
'''

import numpy as np
from gfast.utils import angle_q, dec_to_qary_vec, get_qs, qary_vector_banned


def singleton_detection_noiseless(U_slice, **kwargs): #kunal change this later!
    '''
    Finds the true index of a singleton, or the best-approximation singleton of a multiton.
    Assumes P = n + 1 and D = [0; I].
    
    Arguments
    ---------
    U_slice : numpy.ndarray, (P,).
    The WHT component of a subsampled bin, with element i corresponding to delay i.
    
    Returns
    -------
    k : numpy.ndarray
    Index of the corresponding right node, in binary form.
    '''
    banned_indices_toggle = kwargs.get('banned_indices_toggle')
    if banned_indices_toggle:
        return singleton_detection_noiseless_banned(U_slice, **kwargs)
    q = kwargs.get('q')
    angles = np.angle(U_slice)
    angles = q*(angles[1:] - angles[0])/(2*np.pi)
    angles = angles.round().astype(int) % q
    return angles


def singleton_detection_noiseless_banned(U_slice, **kwargs):
    q = kwargs.get('q')
    n = kwargs.get('n')
    banned_indices = kwargs.get('banned_indices')
    qs = get_qs(q, n, banned_indices=banned_indices)
    angles = np.angle(U_slice)
    angles_ret = np.zeros(n, dtype=int)
    for i, q in enumerate(qs):
        angle = q * (angles[i+1] - angles[0])/(2 * np.pi)
        angle = angle.round().astype(int) % q
        angles_ret[i] = angle
    return angles_ret


def singleton_detection_coded(k, **kwargs):
    '''
    Finds the true index of a singleton, or the best-approximation singleton of a multiton.
    Assumes the Delays matrix is generated by a code, and the syndrome decoder is passed to it.

    Arguments
    ---------
    U_slice : numpy.ndarray, (P,).
    The WHT component of a subsampled bin, with element i corresponding to delay i.

    Returns
    -------
    k : numpy.ndarray
    Index of the corresponding right node, in binary form.
    '''
    decoder = kwargs.get('source_decoder')
    dec = decoder(list(k))
    return np.array(dec[0][0, :], dtype=np.int32)


def singleton_detection_mle(U_slice, **kwargs):
    '''
    Finds the true index of a singleton, or the best-approximation singleton of a multiton, in the presence of noise.
    Uses MLE: looks at the residuals created by peeling off each possible singleton.
    
    Arguments
    ---------
    U_slice : numpy.ndarray, (P,).
    The WHT component of a subsampled bin, with element i corresponding to delay i.

    selection : numpy.ndarray.
    The decimal preimage of the bin index, i.e. the list of potential singletons whose signature under M could be the j of the bin.

    S_slice : numpy.ndarray
    The set of signatures under the delays matrix D associated with each of the elements of 'selection'.

    n : int
    The signal's number of bits.

    Returns
    -------
    k : numpy.ndarray, (n,)
    The index of the singleton.

    '''
    selection, S_slice, q, n, banned_indices_toggle = kwargs.get("selection"), kwargs.get("S_slice"), kwargs.get("q"), kwargs.get("n"), kwargs.get('banned_indices_toggle')
    P = S_slice.shape[0]
    print('selection', selection)
    print('selection S_slice', S_slice)
    alphas = 1/P * np.dot(np.conjugate(S_slice).T, U_slice)
    residuals = np.linalg.norm(U_slice - (alphas * S_slice).T, ord=2, axis=1)
    k_sel = np.argmin(residuals)
    if banned_indices_toggle:
        banned_indices = kwargs.get('banned_indices')
        qs = get_qs(q, n, banned_indices=banned_indices)
        return qary_vector_banned(selection[k_sel], qs)
    return dec_to_qary_vec(selection[k_sel], q, n)



def singleton_detection_nr(U_slice, **kwargs):
    """
    Assumes p1 is the default n+1, which is also the source parity (#rows in D matrix)
    """
    q_max, p1 = kwargs.get("q"), kwargs.get("source_parity")
    banned_indices = kwargs.get('banned_indices')
    if banned_indices is None:
        banned_indices = {}
    qs = get_qs(q_max, p1-1, banned_indices=banned_indices)
    angles = [2 * np.pi / q * np.arange(q + 1) for q in qs]
    U_slice_zero = U_slice[0::p1]
    k_sel_qary = np.zeros((p1-1, ), dtype=int)
    for i in range(1, p1):
        U_slice_i = U_slice[i::p1]
        angle = np.angle(np.mean(U_slice_zero * np.conjugate(U_slice_i))) % (2 * np.pi)
        idx = (np.abs(angles[i-1] - angle)).argmin() % qs[i-1]
        k_sel_qary[i-1] = idx
    return k_sel_qary



def singleton_detection(U_slice, method_source="identity", method_channel="identity", **kwargs):
    """
    Recovers the index value k of a singleton.
    Parameters
    ----------
    U_slice : np.array
    The relevant subsampled fourier transform to be considered

    method_source
    method of reconstruction for source coding: "identity" - default setting, should be used unless you know that all
                                                indicies have low hamming weight
                                                "coded" - Currently only supports prime q, if you know the max hamming
                                                weight of less than t this option should be used and will greatly reduce
                                                complexity. Note a source_decoder object must also be passed

    method_channel
    Method of reconstruction for channel coding: "mle" - exact MLE computation. Fine for small problems but not
                                                         recommended it is exponential in n
                                                 "nr" - symbol-wise recovery suitable when a repetition type code is used
                                                 "identity" - no channel coding, only use when there is no noise

    Returns
    -------
    Value of the computed singleton index k
    """
    # Split detection into two phases, channel and source decoding
    k = {
        "mle": singleton_detection_mle,
        "nr": singleton_detection_nr,
        "identity": singleton_detection_noiseless,
    }.get(method_channel)(U_slice, **kwargs)
    
    if method_source != "identity":
        k = {
            "coded": singleton_detection_coded
        }.get(method_source)(k, **kwargs)
    return k