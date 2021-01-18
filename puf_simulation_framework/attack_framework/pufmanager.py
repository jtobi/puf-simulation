from .pathmanager import PathManager
import anytree
import uuid
import joblib


class PUFManager:

    def __init__(self, base_path):
        self.path_manager = PathManager()
        base_node = anytree.Node('base_path_name')
        tmp_node = anytree.Node('topic_name', parent=base_node)
        tmp_node = anytree.Node('puf_type_name', parent=tmp_node)
        tmp_node = anytree.Node('puf_identifier_name', parent=tmp_node)
        anytree.Node('puf_infos', parent=tmp_node)
        anytree.Node('puf_objects', parent=tmp_node)

        self.path_manager.root_node = base_node
        self.puf_parameters = {}

    def get_puf_info_filename(self, puf_id):
        return 'puf_info_{:s}.dict'.format(puf_id)

    def get_puf_object_filename(self, puf_id):
        return 'puf_object_{:s}.puf'.format(puf_id)

    def create_pufs(self, puf_class, puf_parameters, number_of_pufs):
        self.puf_parameters = puf_parameters

        for current_puf in range(number_of_pufs):
            puf_id = str(uuid.uuid4())[:8]
            puf_info = {'puf_id': puf_id, 'puf_index': current_puf}
            puf_info.update(puf_parameters)

            puf_object = puf_class(**puf_parameters)

            self.path_manager.update_path_content({'puf_type_name': puf_object.get_puf_name(),
                                                   'puf_identifier_name': puf_object.get_puf_identifier()})

            puf_info_filename = self.path_manager.resolve_filename('puf_infos', self.get_puf_info_filename(puf_id))
            puf_object_filename = self.path_manager.resolve_filename('puf_objects', self.get_puf_object_filename(puf_id))

            joblib.dump(puf_info, puf_info_filename)
            joblib.dump(puf_object, puf_object_filename)

    def load_puf_infos(self, puf_type, puf_identifier, num_pufs_to_load=None):
        self.path_manager.update_path_content({'puf_type_name': puf_type,
                                               'puf_identifier_name': puf_identifier})
        # load all dicts
        puf_info_filenames = self.path_manager.list_files_in_path_node('puf_infos')

        puf_infos = []
        for f_name in puf_info_filenames:
            puf_infos.append(joblib.load(f_name))

        return puf_infos

    def load_puf_objects(self, puf_type, puf_identifier, list_puf_ids):
        self.path_manager.update_path_content({'puf_type_name': puf_type,
                                               'puf_identifier_name': puf_identifier})

        puf_objects = []
        for id in list_puf_ids:
            puf_object_filename = self.path_manager.resolve_filename('puf_objects', self.get_puf_object_filename(id))
            puf_objects.append(joblib.load(puf_object_filename))

        return puf_objects






