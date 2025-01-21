import numpy as np
from gfast.gfast import GFAST
from gfast.utils import gwht, dec_to_qary_vec, NpEncoder, qary_vector_banned, get_qs, get_signature
import json
from sklearn.metrics import r2_score


class TestHelper:

    def __init__(self, signal_args, methods, subsampling_args, test_args, exp_dir, subsampling=True):

        self.n = signal_args["n"]
        self.q = signal_args["q"]

        self.exp_dir = exp_dir
        self.subsampling = subsampling

        self.signal_args = signal_args
        self.subsampling_args = subsampling_args
        self.test_args = test_args

        if self.subsampling:
            if len(set(methods).intersection(["gfast"])) > 0:
                self.train_signal = self.load_train_data()
            if len(set(methods).intersection(["gfast_binary"])) > 0:
                self.train_signal_binary = self.load_train_data_binary()
            if len(set(methods).intersection(["lasso"])) > 0:
                self.train_signal_uniform = self.load_train_data_uniform()
            if len(set(methods).intersection(["gfast_coded"])) > 0:
                self.train_signal_coded = self.load_train_data_coded()
            self.test_signal = self.load_test_data()
        else:
            self.train_signal = self.load_full_data()
            self.test_signal = self.train_signal
            if any([m.startswith("binary") for m in methods]):
                raise NotImplementedError  # TODO: implement the conversion


    def generate_signal(self, signal_args):
        raise NotImplementedError


    def load_train_data(self):
        signal_args = self.signal_args.copy()
        query_args = self.subsampling_args.copy()
        query_args.update({
            #KUNAL: I changed query method to simple and delays method channel from nso to identity
            #also changing subsampling method from gfast to input from query args
            "subsampling_method": query_args["query_method"],
            "query_method": query_args["query_method"],
            "delays_method_source": "identity",
            "delays_method_channel": query_args["delays_method_channel"] 
        })
        signal_args["folder"] = self.exp_dir / "train"
        signal_args["query_args"] = query_args
        return self.generate_signal(signal_args)


    def load_train_data_coded(self):
        signal_args = self.signal_args.copy()
        query_args = self.subsampling_args.copy()
        query_args.update({
            "subsampling_method": "gfast",
            "query_method": "complex",
            "delays_method_source": "coded",
            "delays_method_channel": "nso",
            "t": signal_args["t"]
        })
        signal_args["folder"] = self.exp_dir / "train_coded"
        signal_args["query_args"] = query_args
        return self.generate_signal(signal_args)


    def load_train_data_binary(self):
        return None


    def load_train_data_uniform(self):
        signal_args = self.signal_args.copy()
        query_args = self.subsampling_args.copy()
        n_samples = query_args["num_subsample"] * (signal_args["q"] ** query_args["b"]) *\
                    query_args["num_repeat"] * (signal_args["n"] + 1)
        query_args = {"subsampling_method": "uniform", "n_samples": n_samples}
        signal_args["folder"] = self.exp_dir / "train_uniform"
        signal_args["query_args"] = query_args
        return self.generate_signal(signal_args)


    def load_test_data(self):
        signal_args = self.signal_args.copy()
        (self.exp_dir / "test").mkdir(exist_ok=True)
        signal_args["query_args"] = {"subsampling_method": "uniform", "n_samples": self.test_args.get("n_samples")}
        signal_args["folder"] = self.exp_dir / "test"
        signal_args["noise_sd"] = 0
        return self.generate_signal(signal_args)


    def load_full_data(self):
        #   TODO: implement
        return None


    def compute_model(self, method, model_kwargs, report=False, verbosity=0):
        if method == "gwht":
            return self._calculate_gwht(model_kwargs, report, verbosity)
        elif method == "gfast":
            return self._calculate_gfast(model_kwargs, report, verbosity)
        elif method == "gfast_binary":
            return self._calculate_gfast_binary(model_kwargs, report, verbosity)
        elif method == "gfast_coded":
            return self._calculate_gfast_coded(model_kwargs, report, verbosity)
        elif method == "lasso":
            return self._calculate_lasso(model_kwargs, report, verbosity)
        else:
            raise NotImplementedError()


    def test_model(self, method, **kwargs):
        if method == "gfast" or method == "gfast_coded" or method == "lasso":
            if self.test_signal.banned_indices_toggle:
                return self._test_qary_banned(**kwargs)
            else:
                return self._test_qary(**kwargs)
        elif method == "gfast_binary":
            return self._test_binary(**kwargs)
        else:
            raise NotImplementedError()


    def _calculate_gwht(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates GWHT coefficients of the RNA fitness function. This will try to load them
        from the results folder, but will otherwise calculate from scratch. If save=True,
        then coefficients will be saved to the results folder.
        """
        if verbosity >= 1:
            print("Finding all GWHT coefficients")

        beta = gwht(self.train_signal, q=4, n=self.n)
        print("Found GWHT coefficients")
        return beta


    def _calculate_gfast(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates GWHT coefficients of the RNA fitness function using GFAST.
        """
        if verbosity >= 1:
            print("Estimating GWHT coefficients with GFAST")
        gfast = GFAST( #KUNAL: changed reconstruct method channel from nso to identity
            #Changed back to nso for tiger
            reconstruct_method_source="identity",
            reconstruct_method_channel=self.subsampling_args["delays_method_channel"], #CHANGED TO IDENTITY FOR NSO EXPERIMENTS, REMEMBER TO FIX THIS!!!!!!!
            num_subsample=model_kwargs["num_subsample"],
            num_repeat=model_kwargs["num_repeat"],
            b=model_kwargs["b"]
        )
        self.train_signal.noise_sd = model_kwargs["noise_sd"]
        out = gfast.transform(self.train_signal, verbosity=verbosity, timing_verbose=(verbosity >= 1), report=report)
        if verbosity >= 1:
            print("Found GWHT coefficients")
        return out


    def _calculate_gfast_coded(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates GWHT coefficients of the RNA fitness function using GFAST.
        """
        if verbosity >= 1:
            print("Estimating GWHT coefficients with GFAST")

        decoder = get_reed_solomon_dec(self.signal_args["n"], self.signal_args["t"], self.signal_args["q"])
        gfast = GFAST(
            reconstruct_method_source="coded",
            reconstruct_method_channel="nso",
            num_subsample=model_kwargs["num_subsample"],
            num_repeat=model_kwargs["num_repeat"],
            b=model_kwargs["b"],
            source_decoder=decoder
        )
        self.train_signal_coded.noise_sd = model_kwargs["noise_sd"]
        out = gfast.transform(self.train_signal_coded, verbosity=verbosity, timing_verbose=(verbosity >= 1), report=report)
        if verbosity >= 1:
            print("Found GWHT coefficients")
        return out


    def _calculate_gfast_binary(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates GWHT coefficients of the RNA fitness function using GFAST.
        """
        factor = round(np.log(self.q) / np.log(2))

        if verbosity >= 1:
            print("Estimating GWHT coefficients with GFAST")
        gfast = GFAST(
            reconstruct_method_source="identity",
            reconstruct_method_channel="nso",
            num_subsample=model_kwargs["num_subsample"],
            num_repeat=max(1, model_kwargs["num_repeat"] // factor),
            b=factor * model_kwargs["b"],
        )
        self.train_signal_binary.noise_sd = model_kwargs["noise_sd"] / factor
        out = gfast.transform(self.train_signal_binary, verbosity=verbosity, timing_verbose=(verbosity >= 1), report=report)
        if verbosity >= 1:
            print("Found GWHT coefficients")
        return out


    def _calculate_lasso(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates GWHT coefficients of the RNA fitness function using LASSO. This will try to load them
        from the results folder, but will otherwise calculate from scratch. If save=True,
        then coefficients will be saved to the results folder.
        """
        if verbosity > 0:
            print("Finding Fourier coefficients with LASSO")

        self.train_signal_uniform.noise_sd = model_kwargs["noise_sd"]
        out = lasso_decode(self.train_signal_uniform, model_kwargs["n_samples"], noise_sd=model_kwargs["noise_sd"])

        if verbosity > 0:
            print("Found Fourier coefficients")

        return out


    def _test_qary(self, beta):
        """
        :param beta:
        :return:
        """
        if len(beta.keys()) > 0:
            test_signal = self.test_signal.signal_t
            # print("Test signal", test_signal)
            (sample_idx_dec, samples) = list(test_signal.keys()), list(test_signal.values())
            # print("SAMPLES",samples)
            batch_size = 10000
            beta_keys = list(beta.keys())
            beta_values = list(beta.values())

            y_hat = []
            for i in range(0, len(sample_idx_dec), batch_size):
                sample_idx_dec_batch = sample_idx_dec[i:i + batch_size]
                sample_idx_batch = dec_to_qary_vec(sample_idx_dec_batch, self.q, self.n)
                qs = np.array([self.q] * self.n)
                omegas = np.exp(2j * np.pi/qs)
                omegas = omegas.reshape(self.n, 1)
                freqs = np.array(sample_idx_batch).T @ np.array(beta_keys).T
                H = np.exp(2j * np.pi * freqs / self.q)
                y_hat.append(H @ np.array(beta_values))
            y_hat = np.concatenate(y_hat)
            nmse = np.linalg.norm(y_hat - samples) ** 2 / np.linalg.norm(samples) ** 2
            r2 = r2_score(np.real(y_hat), np.real(samples))
            # print(f"NMSE: {nmse}, R2: {r2}")
            return nmse, r2
        else:
            return 1, -1


    def _test_binary(self, beta):
        """
        :param beta:
        :return:
        """
        if len(beta.keys()) > 0:
            test_signal = self.test_signal.signal_t
            (sample_idx_dec, samples) = list(test_signal.keys()), list(test_signal.values())
            batch_size = 10000

            beta_keys = list(beta.keys())
            beta_values = list(beta.values())

            y_hat = []
            for i in range(0, len(sample_idx_dec), batch_size):
                sample_idx_dec_batch = sample_idx_dec[i:i + batch_size]
                sample_idx_batch = dec_to_qary_vec(sample_idx_dec_batch, 2, 2 * self.n)
                freqs = np.array(sample_idx_batch).T @ np.array(beta_keys).T
                H = np.exp(2j * np.pi * freqs / 2)
                y_hat.append(H @ np.array(beta_values))

            # TODO: Write with an if clause
            y_hat = np.abs(np.concatenate(y_hat))

            return np.linalg.norm(y_hat - samples) ** 2 / np.linalg.norm(samples) ** 2
        else:
            return 1


    def _test_qary_banned(self, beta):
        """
        :param beta:
        :return:
        """
        if len(beta.keys()) > 0:
            test_signal = self.test_signal.signal_t
            # print("Test signal", test_signal)
            qs = get_qs(self.q, self.n, banned_indices=self.test_signal.banned_indices)
            (sample_idx_dec, samples) = list(test_signal.keys()), list(test_signal.values())
            # print("SAMPLES/QS", qs, samples)
            batch_size = 10000
            beta_keys = list(beta.keys())
            beta_values = list(beta.values())
            y_hat = []

            for i in range(0, len(sample_idx_dec), batch_size):
                sample_idx_dec_batch = sample_idx_dec[i:i + batch_size]
                sample_idx_batch = np.array([qary_vector_banned(x, qs) for x in sample_idx_dec_batch])
                H = np.empty((np.shape(sample_idx_batch)[0], np.shape(beta_keys)[0]), dtype=complex)
                for i, k in enumerate(np.array(beta_keys)):
                    signature = get_signature(self.q, self.n, sample_idx_batch, k, banned_indices=self.test_signal.banned_indices)
                    H[:, i] = signature.T
                y_hat.append(H @ np.array(beta_values))
            y_hat = np.concatenate(y_hat)
            nmse = np.linalg.norm(y_hat - samples) ** 2 / np.linalg.norm(samples) ** 2
            r2 = r2_score(np.real(y_hat), np.real(samples))
            # print(f"NMSE: {nmse}, R2: {r2}")
            return nmse, r2
        else:
            return 1, -1