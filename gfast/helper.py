from test_helper import TestHelper
from helper_signal import BioSubsampledSignal

class Helper(TestHelper):

    def __init__(self, signal_args, methods, subsampling_args, test_args, exp_dir, subsampling=True):
        signal_args.update({
            "n_samples": test_args.get("n_samples")
        })
        super().__init__(signal_args, methods, subsampling_args, test_args, exp_dir, subsampling)

    def generate_signal(self, signal_args):
        return BioSubsampledSignal(**signal_args)
