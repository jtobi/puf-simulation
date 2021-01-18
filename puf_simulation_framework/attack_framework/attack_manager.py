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


class AttackManager:
    """
    The AttackManager performs an attack (an 'experiment') on a population of pufs. This experiment is identified by
    a unique ID (also called population ID). Each attack consist of several single attacks. Each single attack in turn
    has a unique ID.
    """

    def __init__(self, path_manager: PathManager, enable_logger=True):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        if enable_logger:
            self.logger.addHandler(ch)
        self.logger.info('Init attack manager')

        self.path_manager = path_manager
        base_node = anytree.Node('base_path_name')
        tmp_node = anytree.Node('topic_name', parent=base_node)
        tmp_node = anytree.Node('puf_type_name', parent=tmp_node)
        tmp_node = anytree.Node('puf_identifier_name', parent=tmp_node)
        single_attack_node = anytree.Node('puf_single_attack_folder_name', parent=tmp_node)
        single_attack_node = anytree.Node('puf_datasets_folder', parent=single_attack_node)
        path_manager.root_node = base_node
        # anytree.Node('puf_infos', parent=tmp_node)
        # anytree.Node('puf_objects', parent=tmp_node)

        self.attack_object_list = []

    def perform_single_attack(self, attack_object, model_inputs, model_outputs, attack_parameters, current_puf, puf_wrapper,
                              add_validation_set=False):
        """
        models_inputs and model_outputs are dicts with possible keys: 'training', 'test', 'validation'
        'validation' is optional, the other two are required

        attack_object must have the following functions:
        fit(X_training, Y_training, X_validation, Y_validation) - xxxxx_validation is again optional
        predict(X)
        :param attack_class:
        :param model_inputs:
        :param model_outputs:
        :param add_validation_set:
        :return:
        """

        training_input = model_inputs['training']
        training_output = model_outputs['training']

        max_num_tries = attack_parameters.get('max_num_tries', 1)
        target_accuracy = attack_parameters.get('target_accuracy', 1.0)
        use_all_tries = attack_parameters.get('use_all_tries', False)
        add_puf_to_fit = attack_parameters.get('add_puf_to_fit', False)

        # TODO: refactor validation input as fit parameter?
        additional_fit_parameters = {}
        if add_puf_to_fit:
            additional_fit_parameters['puf_instance'] = current_puf
        additional_fit_parameters.update(attack_parameters)

        current_try = 0
        continue_with_tries = True
        start_time = time.time()
        test_accuracies = []
        fit_log_list = []

        while continue_with_tries:
            if add_validation_set:
                validation_input = model_inputs['validation']
                validation_output = model_outputs['validation']
                fit_log = attack_object.fit(training_input, training_output, validation_input,
                                            validation_output, **additional_fit_parameters)
            else:
                fit_log = attack_object.fit(training_input, training_output, **additional_fit_parameters)
            if fit_log is None:
                fit_log = {}

            fit_log['training_accuracy'] = self.compute_accuracy(attack_object, model_inputs['training'],
                                                                 model_outputs['training'])
            fit_log['test_accuracy'] = self.compute_accuracy(attack_object, model_inputs['test'], model_outputs['test'])
            test_accuracies.append(fit_log['test_accuracy'])

            if not use_all_tries:
                if target_accuracy <= fit_log['test_accuracy']:
                    continue_with_tries = False

            current_try += 1
            if current_try >= max_num_tries:
                continue_with_tries = False

            fit_log_list.append(fit_log)

        duration = time.time() - start_time

        result_dict = {}
        result_dict['best_test_accuracy'] = np.max(test_accuracies)
        result_dict['attack_duration'] = duration
        result_dict['puf_weights'] = current_puf.export_weights()

        result_dict['fit_log_list'] = fit_log_list

        if add_validation_set:
            result_dict['validation_accuracy'] = self.compute_accuracy(attack_object, model_inputs['validation'],
                                                                       model_outputs['validation'])

        return result_dict

    def compute_accuracy(self, model, model_input, expected_output):
        prediction = model.predict(model_input).astype('int')
        prediction = prediction.squeeze()
        return (prediction == expected_output).sum() / prediction.shape[0]

    def create_single_attack_info_name(self):
        return 'single_puf_attack_info_{:s}.jbl'.format(str(uuid.uuid4())[:8])

    def create_population_attack_name(self):
        pop_id = str(uuid.uuid4())[:8]
        return self.get_population_info_filename(pop_id), pop_id

    def get_population_info_filename(self, pop_id):
        return 'population_attack_info_{:s}.jbl'.format(pop_id)

    def store_population_info(self, pop_info, filename):
        self.store_info_dict(pop_info, filename, 'puf_identifier_name')

    def load_population_info(self, filename):
        return self.load_info_dict(filename, 'puf_identifier_name')

    def store_single_attack_info(self, attack_info, filename):
        self.store_info_dict(attack_info, filename, 'puf_single_attack_folder_name')

    def load_single_attack_info(self, filename):
        return self.load_info_dict(filename, 'puf_single_attack_folder_name')

    def store_dataset_for_single_attack(self, attack_id, dataset):
        return self.store_info_dict(dataset, 'dataset_{:s}'.format(attack_id), 'puf_datasets_folder')
    
    def load_dataset_for_single_attack(self, attack_id):
        return self.load_info_dict('dataset_{:s}'.format(attack_id), 'puf_datasets_folder')

    def store_info_dict(self, attack_info, filename, path_node):
        file_path = self.path_manager.resolve_filename(path_node, filename)
        joblib.dump(attack_info, file_path)

    def set_single_attack_info_storage_location(self, population_info_id):
        self.path_manager.update_path_content(
            dict(puf_single_attack_folder_name='single_attack_infos_{:s}'.format(population_info_id)))

    def load_info_dict(self, filename, path_node):
        file_path = self.path_manager.resolve_filename(path_node, filename)
        return joblib.load(os.path.abspath(file_path))

    def attack_puf_population(self, puf_wrapper: PUFWrapper, attack_parameters: dict, attack_object: AttackObject):
        """

        :param puf_wrapper:
        :param attack_parameters:
        :param attack_object:
        :return:
        """
        # Create a dummy PUF that is required to get the PUF type and identifier strings
        puf_wrapper.create_new_instance(do_not_store=True)
        self.path_manager.update_path_content(dict(puf_type_name=puf_wrapper.get_puf_type_name(),
                                                   puf_identifier_name=puf_wrapper.get_puf_parameter_identifier()))

        num_pufs_to_attack = attack_parameters['num_pufs_to_attack']
        add_validation_to_model = attack_parameters['add_validation']

        # add option for static challenges i.e. pass rand state for challenge gen
        population_file_name, population_id = self.create_population_attack_name()
        self.set_single_attack_info_storage_location(population_id)

        population_info = {}
        population_info.update(attack_parameters)

        self.logger.info('Starting attack on population of {:d} PUFs.'.format(num_pufs_to_attack))
        for puf_index in range(num_pufs_to_attack):
            puf_wrapper.create_new_instance()

            inputs, outputs = self.create_inputs_outputs(puf_wrapper, attack_parameters)
            current_attack_object = copy.deepcopy(attack_object)
            self.attack_object_list.append(current_attack_object)
            attack_parameters['puf_index'] = puf_index
            attack_result = self.perform_single_attack(current_attack_object, inputs, outputs, attack_parameters,
                                                       puf_wrapper.current_puf, puf_wrapper,
                                                       add_validation_set=add_validation_to_model)
            self.logger.info(
                'Finished attack on puf #{:d} of {:d}. Best accuracy {:f}'.format(puf_index + 1, num_pufs_to_attack,
                                                                                  attack_result['best_test_accuracy']))
            attack_result['puf_index'] = puf_index
            attack_filename = self.create_single_attack_info_name()
            self.store_single_attack_info(attack_result, attack_filename)

            population_info['population_id'] = population_id
            population_info['num_pufs_attacked_so_far'] = puf_index + 1
            population_info['attack_info'] = attack_object.get_summary_dict()
            population_info['puf_parameters'] = puf_wrapper.puf_parameters
            self.store_population_info(population_info, population_file_name)
        return

    def create_inputs_outputs(self, puf_wrapper, attack_parameters):
        training_set_size = attack_parameters['training_set_size']
        test_set_size = attack_parameters['test_set_size']
        validation_set_size = attack_parameters['validation_set_size']
        perform_reliability_attack = attack_parameters.get('perform_reliability_attack', False)
        reliability_repetitions = attack_parameters.get('reliability_repetitions', 0)
        reliability_type = attack_parameters.get('reliability_type', None)
        training_input, training_output = puf_wrapper.create_input_outputs(training_set_size,
                                                                           add_reliabilty_to_output=perform_reliability_attack,
                                                                           num_repetitions_to_evaluate=reliability_repetitions,
                                                                           reliability_type=reliability_type)
        validation_input, validation_output = puf_wrapper.create_input_outputs(validation_set_size,
                                                                               add_reliabilty_to_output=perform_reliability_attack,
                                                                               num_repetitions_to_evaluate=reliability_repetitions,
                                                                               reliability_type=reliability_type)
        test_input, test_output = puf_wrapper.create_input_outputs(test_set_size,
                                                                   add_reliabilty_to_output=perform_reliability_attack,
                                                                   num_repetitions_to_evaluate=reliability_repetitions,
                                                                   reliability_type=reliability_type)
        inputs = {'training': training_input, 'test': test_input, 'validation': validation_input}
        outputs = {'training': training_output, 'test': test_output, 'validation': validation_output}

        return inputs, outputs
    def get_single_attack_info_filenames(self, population_id):
        self.set_single_attack_info_storage_location(population_id)
        single_attack_path = self.path_manager.resolve_path('puf_single_attack_folder_name')

        pop_info_file_names = os.listdir(single_attack_path)
        pop_info_file_names = [f for f in pop_info_file_names if
                               os.path.isfile(os.path.abspath(os.path.join(single_attack_path, f))) and f.endswith(
                                   '.jbl')]

        if len(pop_info_file_names) == 0:
            raise Exception(
                'No attack infos found for ID {:s} at absolute path {:s}'.format(population_id, os.path.abspath(
                    single_attack_path)))
        return pop_info_file_names

    def load_single_attack_info_by_index(self, population_id, attack_index):
        pop_info_file_names = self.get_single_attack_info_filenames(population_id)
        return self.load_single_attack_info(pop_info_file_names[attack_index])

    def get_number_of_single_attack_infos(self, population_id):
        pop_info_file_names = self.get_single_attack_info_filenames(population_id)
        return len(pop_info_file_names)

    def load_multiple_single_attacks_for_experiment(self, population_id, num_to_load=None, delete_dict_entries=[]):
        pop_info_file_names = self.get_single_attack_info_filenames(population_id)

        if num_to_load is None:
            num_to_load = len(pop_info_file_names)

        single_attack_infos = []
        for file_name in pop_info_file_names[:num_to_load]:
            tmp = self.load_single_attack_info(file_name)
            for key in delete_dict_entries:
                del tmp[key]
            single_attack_infos.append(tmp)

        return single_attack_infos

    def load_all_population_infos(self):
        pop_info_path = self.path_manager.resolve_path('puf_identifier_name')

        pop_info_file_names = [f for f in os.listdir(pop_info_path) if
                               os.path.isfile(os.path.join(pop_info_path, f)) and f.endswith('.jbl')]

        pop_infos = []
        for p in pop_info_file_names:
            pop_infos.append(self.load_population_info(p))

        return pop_infos

    def load_population_info_by_tag(self, tag):
        return self.load_population_info(self.get_population_info_filename(tag))
