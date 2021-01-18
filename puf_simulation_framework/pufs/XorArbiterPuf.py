# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:08:37 2017

@author: Johannes
"""
import numpy as np
from .ArbiterPuf import ArbiterPuf
from .Puf import AbstractPUF
from .puf_util import PufUtil


class XorArbiterPuf(AbstractPUF):
    puf_name = 'classic_xor_arbiter_puf'

    # def __init__(self, num_xors, num_stages, weights_mean=0.0, weights_var=1.0, noise_type='none', error_rate=0.0,
    #              gauss_error_mean=0.0, gauss_error_var=0.0, **kwargs):
    def __init__(self, **kwargs):
        self.num_stages = kwargs['num_stages']
        self.num_xors = kwargs['num_xors']
        self.error_rate = kwargs.get('error_rate', 0.0)
        self.gauss_error_mean = kwargs.get('gauss_error_mean', 0.0)
        self.gauss_error_var = kwargs.get('gauss_error_var', 0.0)
        self.weights_mean = kwargs.get('weights_mean', 0.0)
        self.weights_var = kwargs.get('weights_var', 1.0)
        self.arbiter_pufs = []

        for x in range(self.num_xors):
            self.arbiter_pufs.append(ArbiterPuf(**kwargs))

    def compute_response(self, puf_input, enable_noise=False, input_is_feature_vector=False, **kwargs):
        """Compute puf responses for a matrix of challenges
        
        @param challenges Challenge matrix, size: numOfChallenges X num_stages
        @param enable_noise Allows to disable noise
        
        """
        output_type = kwargs.get('output_type', 'int')
        perform_merge = kwargs.get('merge_individual_outputs', True)

        if output_type == 'int':
            merge_function = np.bitwise_xor
        elif output_type == 'raw' or 'abs_raw':
            merge_function = lambda x, y: np.prod(np.stack([x, y, ], axis=0), axis=0)
        else:
            raise ValueError

        if perform_merge:
            # Each single Puf receives a unique challenge
            if len(puf_input.shape) == 3:
                responses = self.arbiter_pufs[0].compute_response(puf_input[0], enable_noise, input_is_feature_vector,
                                                                  **kwargs)
                for x in range(1, self.num_xors):
                    responses = merge_function(
                        self.arbiter_pufs[x].compute_response(puf_input[x], enable_noise, input_is_feature_vector,
                                                              **kwargs),
                        responses)
            # All Pufs share the same challenge
            elif len(puf_input.shape) == 2:
                responses = self.arbiter_pufs[0].compute_response(puf_input, enable_noise, input_is_feature_vector,
                                                                  **kwargs)
                for x in range(1, self.num_xors):
                    responses = merge_function(
                        self.arbiter_pufs[x].compute_response(puf_input, enable_noise, input_is_feature_vector,
                                                              **kwargs),
                        responses)
            else:
                raise ValueError("Wrong dimension for puf_input.")
        else:
            individual_responses = []
            if len(puf_input.shape) == 3:
                for x in range(0, self.num_xors):
                    individual_responses.append(
                        self.arbiter_pufs[x].compute_response(puf_input[x], enable_noise, input_is_feature_vector,
                                                              **kwargs))
            # All Pufs share the same challenge
            elif len(puf_input.shape) == 2:
                for x in range(0, self.num_xors):
                    individual_responses.append(
                        self.arbiter_pufs[x].compute_response(puf_input, enable_noise, input_is_feature_vector,
                                                              **kwargs))
            else:
                raise ValueError("Wrong dimension for puf_input.")
            responses = np.stack(individual_responses, axis=0)

        return responses

    def get_puf_type_name(self):
        return 'classic_xor_puf'

    def get_puf_parameter_identifier(self):
        return 'xors_{:d}_stages_{:d}'.format(self.num_xors, self.num_stages)

    def create_random_challenges(self, num_challenges, create_feature_vectors=False,
                                 random_state: np.random.RandomState = None):
        raw_challenges = PufUtil.generate_random_challenges(self.num_stages, 1, num_challenges, rand_state=random_state)
        if create_feature_vectors:
            return PufUtil.challenge_to_feature_vector(raw_challenges)
        else:
            return raw_challenges

    def export_weights(self):
        weights = []
        bias = []

        for arb in self.arbiter_pufs:
            weights.append(arb.weights)
            bias.append(arb.weight_bias)

        weights = np.stack(weights, 0)
        bias = np.stack(bias, 0)
        result = {'weights': weights,
                  'bias': bias}

        return result

    def import_weights(self, weight_dict):

        weights = weight_dict['weights']
        bias = weight_dict['bias']

        for index in range(self.num_xors):
            self.arbiter_pufs[index].weights = weights[index]
            self.arbiter_pufs[index].weight_bias = bias[index]
