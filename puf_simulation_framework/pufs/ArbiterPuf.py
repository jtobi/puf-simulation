# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:45:30 2017

@author: Johannes
"""

import numpy as np
import warnings as wn

from .puf_util import PufUtil

class ArbiterPuf:
    puf_name = 'arbiter_puf'
    
    #def __init__(self, num_stages, weights_mean = 0.0, weights_var = 1.0, noise_type = 'none', error_rate = 0.0, gauss_error_mean = 0.0, gauss_error_var = 0.0):
    def __init__(self, **kwargs):

        self.num_stages = kwargs['num_stages']
        self.error_rate = kwargs.get('error_rate', 0.0)
        self.gauss_error_mean = kwargs.get('gauss_error_mean', 0.0)
        self.gauss_error_var = kwargs.get('gauss_error_var', 0.0)
        self.weights_mean = kwargs.get('weights_mean', 0.0)
        self.weights_var = kwargs.get('weights_var', 1.0)

        self.noise_type = kwargs.get('noise_type', None)
        # possible_noise_types = ['none', 'gauss', 'uniform']
        #
        # if kwargs.get('noise_type', 'none') in possible_noise_types:
        #     self.noise_type = kwargs.get('noise_type', 'none')
        # else:
        #     self.errorType = 'none'
        #     wn.warn('ArbiterPuf: Did not recognize error type')
        
        #The puf is described by num_stages+1 weights. We store the last weight
        #in a separate variable because the associated challenge bit is always 
        #equal to one.
        self.weights = np.random.normal(self.weights_mean, np.sqrt(self.weights_var), self.num_stages)
        self.weight_bias = np.random.normal(self.weights_mean, np.sqrt(self.weights_var), 1)


    def compute_response(self, puf_input, enable_noise=False, input_is_feature_vector=False, **kwargs):
        """Compute puf responses for a matrix of challenges
        
        @param challenges Challenge matrix, size: numOfChallenges X num_stages
        @param enable_noise Allows to disable noise
        
        """
        output_type = kwargs.get('output_type', 'int')

        if not input_is_feature_vector:
            feature_vectors = PufUtil.challenge_to_feature_vector(puf_input)
        else:
            feature_vectors = puf_input

        delays = np.dot(feature_vectors, self.weights) + self.weight_bias
        num_challenges = feature_vectors.shape[0]
        
        if enable_noise is True and self.noise_type == 'gauss':
            delays = delays + np.random.normal(self.gauss_error_mean, np.sqrt(self.gauss_error_var), num_challenges)

        if output_type == 'int':
            #responses are binary values
            responses = (np.sign(delays)+1)/2
            responses = responses.astype('int64')

            if enable_noise is True and self.noise_type == 'uniform':
                errors = np.random.binomial(1, self.error_rate, num_challenges).astype('int8')
                responses = np.bitwise_xor(responses, errors)
        elif output_type == 'raw':
            responses = delays
        elif output_type == 'abs_raw':
            responses = np.abs(delays)
        else:
            raise ValueError
            
        return responses

    def get_concatenated_weights(self):
        return np.concatenate([self.weights, self.weight_bias])

    def create_random_challenges(self, num_challenges, create_feature_vectors=False,
                                 random_state: np.random.RandomState = None):
        raw_challenges = PufUtil.generate_random_challenges(self.num_stages, 1, num_challenges, rand_state=random_state)
        if create_feature_vectors:
            return PufUtil.challenge_to_feature_vector(raw_challenges)
        else:
            return raw_challenges

    def get_puf_type_name(self):
        return 'arbiter_puf'

    def get_puf_parameter_identifier(self):
        return 'stages_{:d}'.format(self.num_stages)

            

        
        