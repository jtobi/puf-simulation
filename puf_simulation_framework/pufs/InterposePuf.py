import numpy as np
import copy
from numpy.ma import shape

from .XorArbiterPuf import XorArbiterPuf
from .Puf import AbstractPUF
from .puf_util import PufUtil


class InterposePuf(AbstractPUF):
    puf_name = 'interpose_puf'

    # def __init__(self, num_x_xor_pufs, num_y_xor_pufs, num_stages, weights_mean=0.0, weights_var=1.0, noise_type='none', error_rate=0.0,
    #              gauss_error_mean=0.0, gauss_error_var=0.0, y_pivot=None):
    def __init__(self, **kwargs):
        self.num_stages = kwargs['num_stages']
        self.num_x_xor_pufs = kwargs['num_x_xor_pufs']
        self.num_y_xor_pufs = kwargs['num_y_xor_pufs']
        self.error_rate = kwargs.get('error_rate', 0.0)
        self.gauss_error_mean = kwargs.get('gauss_error_mean', 0.0)
        self.gauss_error_var = kwargs.get('gauss_error_var', 0.0)
        self.weights_mean = kwargs.get('weights_mean', 0.0)
        self.weights_var = kwargs.get('weights_var', 1.0)

        kwargs['num_xors'] = self.num_x_xor_pufs
        self.x_xor_puf = XorArbiterPuf(**kwargs)
        kwargs['num_xors'] = self.num_y_xor_pufs
        kwargs['num_stages'] = self.num_stages + 1
        self.y_xor_puf = XorArbiterPuf(**kwargs)

        if kwargs.get('y_pivot', None) is None:
            self.y_pivot = self.num_stages // 2
        else:
            self.y_pivot = kwargs.get('y_pivot')

    def compute_response(self, puf_input, enable_noise=False, **kwargs):
        """Compute puf responses for a matrix of challenges
        Input must be challenge vector, not a feature vector!

        @param challenges Challenge matrix, size: numOfChallenges X num_stages
        @param enable_noise Allows to disable noise

        """
        #kwargs['input_is_feature_vector'] = False

        x_kwargs = copy.deepcopy(kwargs)
        # always force the output type
        x_kwargs['output_type'] = 'int'
        external_x_response = kwargs.get('external_x_response', None)
        if external_x_response is None:
            x_response = self.x_xor_puf.compute_response(puf_input, enable_noise=enable_noise,
                                                      **x_kwargs)
        else:
            x_response = external_x_response

        # y_challenge = np.zeros([puf_input.shape[0], puf_input.shape[1], puf_input.shape[2]+1])
        #
        # y_challenge[:, :, :self.y_pivot] = puf_input[:, :, :self.y_pivot]
        # y_challenge[:, :, self.y_pivot] = x_response
        # y_challenge[:, :, self.y_pivot+1:] = puf_input[:, :, self.y_pivot:]
        if kwargs['input_is_feature_vector']:
            y_input = np.zeros([puf_input.shape[0], puf_input.shape[1] + 1])
            y_input[:, :(self.y_pivot + 1)] = (1 - 2 * x_response)[:, np.newaxis] * puf_input[:, :self.y_pivot + 1]
            y_input[:, self.y_pivot + 1:] = puf_input[:, self.y_pivot:]
        else:
            y_input = np.zeros([puf_input.shape[0], puf_input.shape[1] + 1])
            y_input[:, :self.y_pivot] = puf_input[:, :self.y_pivot]
            y_input[:, self.y_pivot] = x_response
            y_input[:, self.y_pivot + 1:] = puf_input[:, self.y_pivot:]
        response = self.y_xor_puf.compute_response(y_input, enable_noise, **kwargs)
        if kwargs.get('debug_output', False):
            return response, x_response, y_input
        else:
            return response

    def get_puf_type_name(self):
        return 'interpose_puf'

    def get_puf_parameter_identifier(self):
        return '({:d}, {:d})_stages_{:d}'.format(self.num_x_xor_pufs, self.num_y_xor_pufs, self.num_stages)

    def create_random_challenges(self, num_challenges, create_feature_vectors=False,
                                 random_state: np.random.RandomState = None):
        raw_challenges = PufUtil.generate_random_challenges(self.num_stages, 1, num_challenges, rand_state=random_state)
        if create_feature_vectors:
            # print(
            #     "You're trying to create feature vectors for the Interpose PUF. This is not supported by the model as responses can only be computed for raw challenges.")
            result =  PufUtil.challenge_to_feature_vector(raw_challenges)
        else:
            result = raw_challenges

        return result

    def export_weights(self, **kwargs):
        result = {}
        export_x = kwargs.get('export_x', True)
        export_y = kwargs.get('export_y', True)

        if export_x:
            x_result = self.x_xor_puf.export_weights()
            result.update({'x_weights': x_result['weights'], 'x_bias': x_result['bias']})
        if export_y:
            y_result = self.y_xor_puf.export_weights()
            result.update({'y_weights': y_result['weights'], 'y_bias': y_result['bias']})
        return result

    def import_weights(self, weight_dict, **kwargs):
        import_x = kwargs.get('import_x', True)
        import_y = kwargs.get('import_y', True)
        if import_x:
            x_dict = {'weights': weight_dict['x_weights'],
                      'bias': weight_dict['x_bias']}
            self.x_xor_puf.import_weights(x_dict)
        if import_y:
            y_dict = {'weights': weight_dict['y_weights'],
                  'bias': weight_dict['y_bias']}
            self.y_xor_puf.import_weights(y_dict)
