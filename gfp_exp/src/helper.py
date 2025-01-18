import sys
sys.path.append("..")
from gfast.test_helper import TestHelper
from src.helper_signal import BioSubsampledSignal

class Helper(TestHelper):

    def __init__(self, signal_args, methods, subsampling_args, test_args, exp_dir, subsampling=True):
        super().__init__(signal_args, methods, subsampling_args, test_args, exp_dir, subsampling)

    def generate_signal(self, signal_args):
        return BioSubsampledSignal(**signal_args)
