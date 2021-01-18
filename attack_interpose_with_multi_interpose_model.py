from puf_simulation_framework.attack_framework import AttackManager
from puf_simulation_framework.attack_framework import PathManager
from puf_simulation_framework.pufs import PUFWrapper, XorArbiterPuf, InterposePuf
from puf_simulation_framework.pytorch_gradient_attack import MultiPytorchWrapper, XorArbiterNet, MultiInterposePufNet
import torch


torch.set_num_threads(1)

puf_parameter = {'num_stages': 64,
                 'num_x_xor_pufs': 3,
                 'num_y_xor_pufs': 2,
                 'create_feature_vectors': True,
                 'y_pivot': 32}
puf_wrapper = PUFWrapper(puf_class=InterposePuf, puf_parameters=puf_parameter)
model_parameters = {'num_stages': puf_parameter['num_stages'],
                    'num_x_xors': puf_parameter['num_x_xor_pufs'],
                    'num_y_xors': puf_parameter['num_y_xor_pufs'],
                    'y_pivot': puf_parameter['y_pivot'],
                    'num_multi_pufs': 4,
                    'input_is_feature_vector': True,
                    'include_x_batch_norm':False,
                    'x_output_type': 'sigmoid2'}

optim_parameters = {'optimizer_name': 'Adam', 'optimizer_parameters': {}}
#optim_parameters = {'optimizer_name': 'Rprop', 'optimizer_parameters': {}}
fit_parameters = {'batch_size': 128,
                  'num_epochs': 20,
                  'verbose': True}
attack_parameters = {'num_pufs_to_attack': 1,
                     'training_set_size': 10000,
                     'test_set_size': 10000,
                     'validation_set_size': 2000,
                     'add_validation': True,
                     'convert_input_to_feature_vector': False,
                     'add_puf_to_fit': True}

pytorch_attack_wrapper = MultiPytorchWrapper(MultiInterposePufNet, model_parameters, optim_parameters, fit_parameters)



path_manager = PathManager()
path_manager.update_path_content(dict(base_path_name='./puf_experiment_results',
                                      topic_name='test_runs'))
attack_manager = AttackManager(path_manager)
attack_manager.attack_puf_population(puf_wrapper, attack_parameters, pytorch_attack_wrapper)