import time
from puf_simulation_framework.pufs import PUFWrapper
from .pathmanager import PathManager
from .attack_object import AttackObject
import uuid
import joblib
import anytree
import numpy as np
import logging
import os
import copy

from .attack_manager import AttackManager


class ReliabilityAttackManager(AttackManager):

    def perform_single_attack(self, attack_object, model_inputs, model_outputs, attack_parameters, current_puf,
                              puf_wrapper, add_validation_set=False):
        """
        Shortened for special case of reliability attack
        :param attack_class:
        :param model_inputs:
        :param model_outputs:
        :param add_validation_set:
        :return:
        """

        training_input = model_inputs['training_noisy']
        training_output = model_outputs['training_noisy']

        add_puf_to_fit = attack_parameters.get('add_puf_to_fit', False)
        store_data_sets = attack_parameters.get('data_sets_to_store', [])

        reliability_estimates = self.estimate_reliability(puf_wrapper, model_inputs, model_outputs, attack_parameters)

        # TODO: refactor validation input as fit parameter?
        additional_fit_parameters = {}
        if add_puf_to_fit:
            additional_fit_parameters['puf_instance'] = current_puf
            additional_fit_parameters['puf_index'] = attack_parameters['puf_index']
        additional_fit_parameters.update(attack_parameters)

        start_time = time.time()
        print(
            'Starting attack on puf with estimated val noisy to noisy acc {:.2f} and val noisy to noise free acc {:.2f}'.format(
                reliability_estimates['noisy_to_noisy_validation'],
                reliability_estimates['noisy_to_noise_free_validation']))

        if add_validation_set:
            validation_input = model_inputs['validation_noisy']
            validation_output = model_outputs['validation_noisy']
            fit_log = attack_object.fit(training_input, training_output, validation_input,
                                        validation_output, **additional_fit_parameters)
        else:
            fit_log = attack_object.fit(training_input, training_output, **additional_fit_parameters)
        duration = time.time() - start_time

        best_noise_free_test_accuracy = self.compute_accuracy(attack_object, model_inputs['test_noisy'],
                                                              model_outputs['test_noise_free'])
        best_noisy_test_accuracy = self.compute_accuracy(attack_object, model_inputs['test_noisy'],
                                                         model_outputs['test_noisy'])

        result_dict = {}
        result_dict['best_noise_free_test_accuracy'] = best_noise_free_test_accuracy
        result_dict['best_noisy_test_accuracy'] = best_noisy_test_accuracy
        result_dict['best_test_accuracy'] = best_noisy_test_accuracy
        result_dict['attack_duration'] = duration
        result_dict['fit_log'] = fit_log
        result_dict['model_inputs'] = model_inputs
        result_dict['model_outputs'] = model_outputs
        result_dict['puf_weights'] = current_puf.export_weights()
        result_dict['reliablity_estimates'] = reliability_estimates
        return result_dict

    def create_inputs_outputs(self, puf_wrapper, attack_parameters):
        training_set_size = attack_parameters['training_set_size']
        test_set_size = attack_parameters['test_set_size']
        validation_set_size = attack_parameters['validation_set_size']
        perform_reliability_attack = attack_parameters.get('perform_reliability_attack', False)
        reliability_repetitions = attack_parameters.get('reliability_repetitions', 0)
        reliability_type = attack_parameters.get('reliability_type', None)

        set_names = ['training_noisy', 'validation_noisy', 'test_noisy',
                     'training_noise_free', 'validation_noise_free', 'test_noise_free']
        set_sizes = [training_set_size, validation_set_size, test_set_size] * 2
        disable_noise = [False, False, False, True, True, True]
        use_challenges = [None, None, None, 'training_noisy', 'validation_noisy', 'test_noisy']
        result_inputs = {}
        result_outputs = {}
        for c_set_name, c_set_size, c_disable_noise, c_use_challenges in zip(set_names, set_sizes, disable_noise,
                                                                             use_challenges):
            if c_use_challenges is not None:
                input = result_inputs[c_use_challenges]
            else:
                input = None
            input, output = puf_wrapper.create_input_outputs(c_set_size,
                                                             add_reliabilty_to_output=perform_reliability_attack,
                                                             num_repetitions_to_evaluate=reliability_repetitions,
                                                             reliability_type=reliability_type,
                                                             disable_noise_for_responses=c_disable_noise,
                                                             challenges=input)
            result_inputs[c_set_name] = input
            result_outputs[c_set_name] = output

        return result_inputs, result_outputs

    def compute_in_out_accuracy(self, responses, expected_respones):
        return np.mean(responses == expected_respones)

    def estimate_reliability(self, puf_wrapper, inputs, outputs, attack_parameters):
        validation_input = inputs['validation_noisy']
        test_input = inputs['test_noisy']
        # Extract response, not reliability information, from outputs
        validation_noisy_responses = outputs['validation_noisy'][:,0]
        validation_noise_free = outputs['validation_noise_free'][:,0]
        test_noisy_responses = outputs['test_noisy'][:,0]
        test_noise_free = outputs['test_noise_free'][:,0]

        # noisy to noisy Validation
        _, second_valiation_noisy_responses = puf_wrapper.create_input_outputs(0,
                                                                            add_reliabilty_to_output=False,
                                                                            challenges=validation_input)
        noisy_to_noisy_validation_acc = self.compute_in_out_accuracy(validation_noisy_responses,
                                                                     second_valiation_noisy_responses)
        noisy_to_noise_free_validation_acc = self.compute_in_out_accuracy(validation_noisy_responses,
                                                                          validation_noise_free)
        _, second_test_noisy_responses = puf_wrapper.create_input_outputs(0,
                                                                       add_reliabilty_to_output=False,
                                                                       challenges=test_input)
        noisy_to_noisy_test_acc = self.compute_in_out_accuracy(test_noisy_responses,
                                                               second_test_noisy_responses)
        noisy_to_noise_free_test_acc = self.compute_in_out_accuracy(test_noisy_responses,
                                                                    test_noise_free)

        estimates = {'noisy_to_noisy_validation': noisy_to_noisy_validation_acc,
                     'noisy_to_noise_free_validation': noisy_to_noise_free_validation_acc,
                     'noisy_to_noisy_free_test': noisy_to_noisy_test_acc,
                     'noisy_to_noise_free_test': noisy_to_noise_free_test_acc}

        return estimates


    def compute_accuracy(self, model, model_input, expected_output):
        prediction = model.predict(model_input).astype('int')
        prediction = prediction.squeeze()
        if len(expected_output.shape) == 2:
            expected_output = expected_output[:,0]
        return (prediction == expected_output).sum() / prediction.shape[0]
