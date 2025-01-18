from gfast.utils import qary_ints, qary_vec_to_dec, gwht, load_data, save_data, qary_ints_banned, get_qs, ftft, find_matrix_indices, qary_vector_banned, save_data4
from gfast.input_signal import Signal
from gfast.query import get_Ms_and_Ds, get_D_identity
from pathlib import Path
from math import floor     
from tqdm import tqdm
import numpy as np
import random
import time
np.random.seed(42)

class SubsampledSignal(Signal):
    """
    A shell Class for input signal/functions that are too large and cannot be stored in their entirety. In addition to
    the signal itself, this must also contain information about the M and D matricies that are used for subsampling
    Notable attributes are included below.

    Attributes
    ---------
    query_args : dict
    These are the parameters that determine the structure of the Ms and Ds needed for subsampling.
    It contains the following sub-parameters:
        b : int
        The max dimension of subsampling (i.e., we will subsample functions with b inputs, or equivalently a signal of
        length q^b)
        all_bs : list, (optional)
        List of all the b values that should be subsampled. This is most useful when you want to repeat an experiment
        many times with different values of b to see which is most efficient
        For a description of the "delays_method_channel", "delays_method_source", "num_repeat" and "num_subsample", see
        the docstring of the GFAST class.
        subsampling_method
            If set to "simple" the M matricies are generated according to the construction in Appendix C, i.e., a
            block-wise identity structure.
            If set to "complex" the elements of the M matricies are uniformly populated from integers from 0 to q-1.
            It should be noted that these matricies are not checked to be full rank (w.r.t. the module where arithemtic is
            over the integer quotient ring), and so it is possible that the actual dimension of subsampling may be
            lower. For large enough n and b this isn't a problem, since w.h.p. the matricies are full rank.

    L : np.array
    An array that enumerates all q^b q-ary vectors of length b

    foldername : str
    If set, and the file {foldername}/Ms_and_Ds.pickle exists, the Ms and Ds are read directly from the file.
    Furthermore, if the transforms for all the bs are in {foldername}/transforms/U{i}_b{b}.pickle, the transforms can be
    directly loaded into memory.
    """
    def _set_params(self, **kwargs):
        self.n = kwargs.get("n")
        self.q = kwargs.get("q")
        self.banned_indices_toggle = kwargs.get('banned_indices_toggle')
        if self.banned_indices_toggle:
            self.banned_indices = kwargs.get('banned_indices')
            self.N = np.prod(get_qs(self.q, self.n, banned_indices=self.banned_indices), dtype=object) #np.prod(get_qs(self.q, self.n, banned_indices=self.banned_indices))
        else:
            self.banned_indices = {}
            self.N = self.q ** self.n
        self.signal_w = kwargs.get("signal_w")
        self.query_args = kwargs.get("query_args")
        self.b = self.query_args.get("b")
        self.all_bs = self.query_args.get("all_bs", [self.b])   # all b values to sample/transform at
        self.num_subsample = self.query_args.get("num_subsample")
        if "num_repeat" not in self.query_args:
            self.query_args["num_repeat"] = 1
        if "train_samples" not in self.query_args:
            self.train_samples = "simple"
        else:
            self.train_samples = self.query_args.get("train_samples")
        self.num_repeat = self.query_args.get("num_repeat")
        self.subsampling_method = self.query_args.get("subsampling_method")
        self.delays_method_source = self.query_args.get("delays_method_source")
        self.delays_method_channel = self.query_args.get("delays_method_channel")
        self.L = None  # List of all length b qary vectors
        self.foldername = kwargs.get("folder")
        self.num_delays = self.query_args.get("num_delays")
        self.query_method = self.query_args.get("query_method")
        self.noise_sd = kwargs.get("noise_sd")
    def _init_signal(self):
        if self.subsampling_method == "uniform":
            self._subsample_uniform()
        elif self.subsampling_method == "generate_samples":
            self._set_Ms_and_Ds_gfast()
            self._generate_train_subsample()
            self._generate_test_subsample()
            exit()
        elif self.subsampling_method == "gfast":
            self._set_Ms_and_Ds_gfast()
            self._subsample_gfast()
        else:
            self._set_Ms_and_Ds_gfast()
            self._subsample_gfast()
    
    def _check_transforms_gfast(self):
        """
        Returns
        -------
        True if the transform is already computed and saved for all values of b, else False
        """
        if self.foldername:
            Path(f"{self.foldername}/transforms/").mkdir(exist_ok=True)
            for b in self.all_bs:
                for i in range(len(self.Ms)):
                    Us_path = Path(f"{self.foldername}/transforms/U{i}_b{b}.pickle")
                    if not Us_path.is_file():
                        return False
            return True
        else:
            return False

    def _set_Ms_and_Ds_gfast(self):
        """
        Sets the values of Ms and Ds, either by loading from folder if exists, otherwise it loaded from query_args
        """
        if self.foldername:
            Path(f"{self.foldername}").mkdir(exist_ok=True)
            Ms_and_Ds_path = Path(f"{self.foldername}/Ms_and_Ds.pickle")
            if Ms_and_Ds_path.is_file():
                self.Ms, self.Ds = load_data(Ms_and_Ds_path)
            else:
                self.Ms, self.Ds = get_Ms_and_Ds(self.n, self.q, **self.query_args)
                save_data((self.Ms, self.Ds), f"{self.foldername}/Ms_and_Ds.pickle")
        else:
            if self.banned_indices_toggle:
                qs = get_qs(self.q, self.n, banned_indices=self.banned_indices)
                self.Ms, self.Ds = get_Ms_and_Ds(self.n, self.q, qs=qs, **self.query_args)
            else:
                self.Ms, self.Ds = get_Ms_and_Ds(self.n, self.q, **self.query_args)

    def _subsample_gfast(self, **kwargs):
        """
        Subsamples and computes the sparse fourier transform for each subsampling group if the samples are not already
        present in the folder
        """
        self.Us = [[{} for j in range(len(self.Ds[i]))] for i in range(len(self.Ms))]
        self.transformTimes = [[{} for j in range(len(self.Ds[i]))] for i in range(len(self.Ms))]
        self.qs_subset = []
        if self.foldername:
            Path(f"{self.foldername}/samples").mkdir(exist_ok=True)
            Path(f"{self.foldername}/transforms/").mkdir(exist_ok=True)

        total_qs = get_qs(self.q, self.n, self.banned_indices)
        pbar = tqdm(total=0, position=0)
        for i in range(len(self.Ms)):
            if self.query_method == 'simple':
                qs_subset = total_qs[len(total_qs) - (i + 1) * self.b : len(total_qs) - i * self.b]
                self.qs_subset.append(qs_subset)
            elif self.query_method == 'complex':
                start_index = (len(total_qs) - (i + 1) * self.b) % len(total_qs) 
                end_index = (len(total_qs) - i * self.b) % len(total_qs)
                if start_index < end_index:
                    qs_subset = total_qs[start_index:end_index]
                else:
                    # This is the wrap-around case
                    qs_subset = np.concatenate((total_qs[start_index:], total_qs[:end_index]))
                self.qs_subset.append(qs_subset)
            else:
                raise NotImplementedError("Query method not simple or complex")
            for j in range(len(self.Ds[i])):
                transform_file = Path(f"{self.foldername}/transforms/U{i}_{j}.pickle")
                if self.foldername and transform_file.is_file():
                    self.Us[i][j], self.transformTimes[i][j] = load_data(transform_file)
                    pbar.total = len(self.Ms) * len(self.Ds[0]) * len(self.Us[i][j])
                    pbar.update(len(self.Us[i][j]))
                else:
                    sample_file = Path(f"{self.foldername}/samples/M{i}_D{j}.pickle")
                    if self.foldername and sample_file.is_file():
                        samples = load_data(sample_file)
                        pbar.total = len(self.Ms) * len(self.Ds[0]) * len(samples)
                        pbar.update(len(samples))
                    else:
                        #assuming M is simple
                        if self.banned_indices_toggle:
                            query_indices = self.get_gfast_banned_query_indices(self.Ms[i], self.Ds[i][j], qs_subset)
                        else:
                            query_indices = self._get_gfast_query_indices(self.Ms[i], self.Ds[i][j])
                        block_length = len(query_indices[0])
                        samples = np.zeros((len(query_indices), block_length), dtype=complex)
                        pbar.total = len(self.Ms) * len(self.Ds[0]) * len(query_indices)
                        if block_length > 10000:
                            for k in range(len(query_indices)):
                                samples[k] = self.subsample(query_indices[k])
                                pbar.update()
                        else:
                            all_query_indices = np.concatenate(query_indices)
                            all_samples = self.subsample(all_query_indices)
                            for k in range(len(query_indices)):
                                samples[k] = all_samples[k * block_length: (k+1) * block_length]
                                pbar.update()
                        if self.foldername:
                            save_data(samples, sample_file)
                        
                    for b in self.all_bs:
                        start_time = time.time() 
                        if self.banned_indices_toggle:
                            self.Us[i][j][b] = self._compute_subtransform_banned(samples, b, qs_subset)
                        else:
                            self.Us[i][j][b] = self._compute_subtransform(samples, b)
                        self.transformTimes[i][j][b] = time.time() - start_time
                    if self.foldername:
                        save_data((self.Us[i][j], self.transformTimes[i][j]), transform_file)

        
    def _generate_train_subsample(self):
        """
        Get all train sample indices and save them out 
        """
        total_qs = get_qs(self.q, self.n, self.banned_indices)
        self.Us = [[{} for j in range(len(self.Ds[i]))] for i in range(len(self.Ms))]
        self.transformTimes = [[{} for j in range(len(self.Ds[i]))] for i in range(len(self.Ms))]
        global newrun 
        newrun = False

        if self.foldername:
            Path(f"{self.foldername}/samples").mkdir(exist_ok=True)
            Path(f"{self.foldername}/transforms/").mkdir(exist_ok=True)

        """
        Extract indices
        """
        for i in range(len(self.Ms)):
            for j in range(len(self.Ds[i])):
                if self.train_samples == 'simple':
                    qs_subset = total_qs[len(total_qs) - (i + 1) * self.b : len(total_qs) - i * self.b]
                elif self.train_samples == 'complex':
                    start_index = (len(total_qs) - (i + 1) * self.b) % len(total_qs) 
                    end_index = (len(total_qs) - i * self.b) % len(total_qs)
                    if start_index < end_index:
                        qs_subset = total_qs[start_index:end_index]
                    else:
                        # This is the wrap-around case
                        qs_subset = np.concatenate((total_qs[start_index:], total_qs[:end_index]))
                sample_file_indices = Path(f"{self.foldername}/samples/M{i}_D{j}_queryindices.pickle")
                sample_file_qaryindices = Path(f"{self.foldername}/samples/M{i}_D{j}_qaryindices.pickle")
                if self.foldername and not sample_file_indices.is_file():
                    query_indices = self.get_gfast_banned_query_indices(self.Ms[i], self.Ds[i][j], qs_subset)
                    random_samples = np.array(qary_vector_banned(query_indices, total_qs))
                    save_data4(random_samples, sample_file_indices)
                    save_data4(query_indices, sample_file_qaryindices)


    def _generate_test_subsample(self):
        """
        Get all test sample indices and save them out 
        """
        total_qs = get_qs(self.q, self.n, self.banned_indices)
        test_file_indices = Path(f"{self.foldername}/../test/signal_t_queryindices.pickle")
        test_file_qaryindices = Path(f"{self.foldername}/../test/signal_t_query_qaryindices.pickle")
        query_indices = self._get_random_query_indices(10000)
        save_data4(query_indices, test_file_qaryindices)
        random_samples = np.array(qary_vector_banned(query_indices, total_qs)).T
        save_data4(random_samples, test_file_indices)


    def _subsample_uniform(self):
        """
        Uniformly subsamples the signal. Useful when you are solving via LASSO
        """
        if self.foldername:
            Path(f"{self.foldername}").mkdir(exist_ok=True)

        sample_file = Path(f"{self.foldername}/signal_t.pickle")
        if self.foldername and sample_file.is_file():
            signal_t = load_data(sample_file)
        else:
            query_indices = self._get_random_query_indices(self.query_args["n_samples"])
            samples = self.subsample(query_indices)
            signal_t = dict(zip(query_indices, samples))
            qs = get_qs(self.q, self.n, banned_indices=self.banned_indices)
            query_indices_qary_batch = np.array([qary_vector_banned(x, qs) for x in query_indices])
            if self.foldername:
                save_data(signal_t, sample_file)
        self.signal_t = signal_t


    def get_all_qary_vectors(self):
        if self.L is None:
            self.L = np.array(qary_ints(self.b, self.q))  # List of all length b qary vectors
        return self.L


    def subsample(self, query_indices):
        raise NotImplementedError


    def banned_subsample(self, query_indices):
        raise NotImplementedError


    def _get_gfast_query_indices(self, M, D_sub):
        """
        Gets the indicies to be queried for a given M and D

        Parameters
        ----------
        M
        D_sub

        Returns
        -------
        base_inds_dec : list
        The i-th element in the list is the affine space {Mx + d_i, forall x}, but in a decimal index, because it is
        more efficient, where d_i is the i-th row of D_sub.
        """
        b = M.shape[1]
        L = self.get_all_qary_vectors()

        """
        Vectorized implementation
        np.shape(base_indices[i]) is n x q**b
        np.shape(base_indices[i]) is p x n x q**b
        """
        ML = (M @ L) % self.q
        base_inds = [(ML + np.outer(d, np.ones(self.q ** b, dtype=int))) % self.q for d in D_sub] 
        base_inds = np.array(base_inds)
        base_inds_dec = []
        for i in range(len(base_inds)):
            base_inds_dec.append(qary_vec_to_dec(base_inds[i], self.q))

        return base_inds_dec
    

    def get_gfast_banned_query_indices(self, M, D_sub, qs):
        b = M.shape[1]
        qs_full = get_qs(self.q, self.n, banned_indices= self.banned_indices)
        qs_full = qs_full[:, np.newaxis]
        L_banned = qary_ints_banned(qs)  

        """
        Vectorized implementation of banned indices removal
        np.shape(base_indices[i]) is n x np.prod(qs_full)
        np.shape(base_indices[i]) is p x n x np.prod(qs_full)
        """
        ML_banned = (M @ L_banned) % qs_full
        base_inds_banned = [(ML_banned + np.outer(d, np.ones(np.prod(qs), dtype=int))) % qs_full for d in D_sub]
        base_inds_banned = np.array(base_inds_banned)
        for i, q in enumerate(qs_full):
            base_inds_banned[:,i,:] = base_inds_banned[:,i,:] % q
        base_inds_dec_banned = []
        for i in range(self.n + 1):
            qs_full = qs_full.flatten()
            base_inds_dec_banned.append(np.array(find_matrix_indices(base_inds_banned[i], qs_full), dtype=object))
        return base_inds_dec_banned


    def _get_random_query_indices(self, n_samples):
        random.seed(42) # Defined so multiple calls of the function will use the same samples
        """
        Returns random indicies to be sampled.

        Parameters
        ----------
        n_samples

        Returns
        -------
        base_ids_dec
        Indicies to be queried in decimal representation
        """
        base_inds_dec = [floor(random.uniform(0, 1) * self.N) for _ in range(n_samples)]
        base_inds_dec = np.unique(np.array(base_inds_dec, dtype=object))
        return base_inds_dec


    def get_MDU(self, ret_num_subsample, ret_num_repeat, b, trans_times=False):
        """
        Allows the GFAST Class to get the effective Ms, Ds and Us (subsampled transforms).
        Parameters
        ----------
        ret_num_subsample
        ret_num_repeat
        b
        trans_times

        Returns
        -------
        Ms_ret
        Ds_ret
        Us_ret
        """
        Ms_ret = []
        Ds_ret = []
        Us_ret = []
        Ts_ret = []
        if ret_num_subsample <= self.num_subsample and ret_num_repeat <= self.num_repeat and b <= self.b:
            subsample_idx = np.random.choice(self.num_subsample, ret_num_subsample, replace=False)
            delay_idx = np.random.choice(self.num_repeat, ret_num_repeat, replace=False)
            for i in subsample_idx:
                Ms_ret.append(self.Ms[i][:, :b])
                Ds_ret.append([])
                Us_ret.append([])
                Ts_ret.append([])
                for j in delay_idx:
                    Ds_ret[-1].append(self.Ds[i][j])
                    Us_ret[-1].append(self.Us[i][j][b])
                    Ts_ret[-1].append(self.transformTimes[i][j][b])

            if trans_times: 
                return Ms_ret, Ds_ret, Us_ret, Ts_ret, subsample_idx
            else:
                return Ms_ret, Ds_ret, Us_ret, subsample_idx
        else:
            raise ValueError("There are not enough Ms or Ds.")
        

    def _compute_subtransform(self, samples, b):
        transform = [gwht(row[::(self.q ** (self.b - b))], self.q, b) for row in samples]
        return transform


    def _compute_subtransform_banned(self, samples, b, qs): 
        transform = [ftft(row[::(self.q ** (self.b - b))], self.q, self.b, qs) for row in samples]
        transform = np.array(transform, dtype=complex)
        pad_boundary = np.prod(qs)
        padded_transform = np.zeros((np.shape(transform)[0], self.q ** self.b), dtype=complex)
        alpha = (self.q ** self.b) - pad_boundary
        padded_transform[:, alpha:] = transform
        return padded_transform 


    def get_source_parity(self):
        return self.Ds[0][0].shape[0]
