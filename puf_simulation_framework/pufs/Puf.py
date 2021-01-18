from abc import ABC, abstractmethod
import numpy as np


class AbstractPUF(ABC):
    """
    abstract PUF class
    """
    puf_name = 'abstract_puf'

    @abstractmethod
    def compute_response(self, puf_input, enable_noise=False, **kwargs):
        pass

    def compute_reliability_response(self, puf_input, num_repetitions_to_evaluate, **kwargs):
        reliability_type = kwargs.get('reliability_type', 'simple_mean')
        raw_responses = np.zeros((num_repetitions_to_evaluate, puf_input.shape[0]))

        for current_repetition in range(num_repetitions_to_evaluate):
            raw_responses[current_repetition, :] = self.compute_response(puf_input, enable_noise=True, **kwargs)

        mean_output = raw_responses.mean(axis=0)
        if reliability_type == 'simple_mean':
            output = mean_output
        elif reliability_type == 'minus_abs':
            output = (abs(0.5-mean_output) + 0.5)
        else:
            raise ValueError()
        return output

    def get_puf_name(self):
        return self.puf_name

    @abstractmethod
    def get_puf_type_name(self):
        """
        Return a string that describes the puf architecture, such as classic_xor_puf
        :return:
        """
        pass

    @abstractmethod
    def get_puf_parameter_identifier(self):
        """
        Return a string that identifies a PUF based on its parameters, such as XOR PUF-> 3_Xors__64_Stages
        :return:
        """
        pass

    # abstractmethod
    #
    # def create_random_challenges(self, num_challenges, create_feature_vectors=False,
    #                              random_state: np.random.RandomState = None):
    #     pass
