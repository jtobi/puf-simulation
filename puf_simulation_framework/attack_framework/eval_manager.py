import numpy as np
from .pathmanager import PathManager
from .attack_manager import AttackManager
import os
import scipy.io as scio






class EvalManager:


    def __init__(self, path_manager:PathManager, attack_manager: AttackManager):
        self.path_manager = path_manager
        self.attack_manager = attack_manager


    def create_population_info_overview(self, puf_type, puf_identifier):
        self.path_manager.update_path_content(dict( puf_type_name=puf_type,
                                              puf_identifier_name=puf_identifier))

        attack_manager = AttackManager(self.path_manager)

        pop_infos = attack_manager.load_all_population_infos()

        for p_info in pop_infos:
            for key, value in p_info.items():
                print(key + ' ' + str(value))


    def export_single_experiment_to_matlab(self, puf_type, puf_identifier, result_path):
        self.path_manager.update_path_content(dict(puf_type_name=puf_type,
                                                   puf_identifier_name=puf_identifier))

        attack_manager = AttackManager(self.path_manager)

        pop_infos = attack_manager.load_all_population_infos()

        for p_info in pop_infos:

            single_infos = attack_manager.load_multiple_single_attacks_for_experiment(p_info['population_id'])
            p_info['single_attack_results'] = single_infos
            p_info['puf_type'] = puf_type
            p_info['puf_identifier'] = puf_identifier

        result_dict = {}
        for p_info in pop_infos:
            result_dict['experiment_{:s}'.format( p_info['population_id'])] = p_info

        if not os.path.isdir(result_path):
            os.makedirs(result_path)
        scio.savemat(os.path.join(result_path, '{:s}_{:s}.mat'.format(puf_type, puf_identifier)),result_dict )



        print()


