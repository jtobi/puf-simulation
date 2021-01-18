import anytree
import os


class PathManager:
    """
    The PathManager takes care of keeping track of the path structure of stored data. The data set of single experiments
    can require different evaluations and usually one works on multiple experiments. A managed folder structure helps
    organizing all results and allows to flexibly add new evaluations etc.

    The path manager assumes the following base structure:
    base_path/topic_name/db_name/collection_name/staging_name/comp_name/eval/eval_name
    Each part of the abstract path above is referred to as a path_node. The path_nodes are filled with path_content
    which allows us to resolve an actual path for the file system.
    An example would be:
    vpor_data/machine_learning/ml_database/collection__2018_01_0__14_55/sin_cos_staging/neural_network_learner/eval/learner_results

    Please note that there are two kinds of path_nodes, those that end with '_name' and those that don't.
    Only the former are resolved with their path_concent. The path_node 'eval', for example, is not resolved.

    The path hierarchy can be dynamically changed or sub-classes can be created that create their own default hierarchy.


    """

    def __init__(self):
        self.path_content = {}
        self.root_node = self.build_default_path_hierachy()

    def build_default_path_hierachy(self):
        # convention: if a node ends with _name it has to be replaced with a look-up value at path building time
        base_node = anytree.Node('base_path_name')
        tmp_node = anytree.Node('topic_name', parent=base_node)
        tmp_node = anytree.Node('puf_type_name', parent=tmp_node)
        tmp_node = anytree.Node('puf_identifier_name', parent=tmp_node)
        anytree.Node('puf_infos', parent=tmp_node)
        anytree.Node('puf_objects', parent=tmp_node)


        # coll_node = anytree.Node('collection_name', parent=tmp_node)
        # tmp_data_node = anytree.Node('tmp_data', parent=coll_node)
        # staging_node = anytree.Node('staging_name', parent=coll_node)
        # tmp_node = anytree.Node('comp_name', parent=staging_node)
        # tmp_node = anytree.Node('eval', parent=staging_node)
        # tmp_node = anytree.Node('eval_name', parent=tmp_node)
        # tmp_node = anytree.Node('figures', parent=tmp_node)
        # tmp_node = anytree.Node('figures_sub_name', parent=tmp_node)

        return base_node

    def update_path_content(self, update_dict):
        """
        Updates the path_content to allow resolving actual paths.
        :param update_dict:
        :return:
        """
        self.path_content.update(update_dict)

    def resolve_path(self, path_node, node_content='', create_path_if_not_found=True):
        """
        Resolve to an actual file_system path. This requires prior setting of the path_content.

        :param path_node: The deepest level to which you want to resolve your path
        :param node_content: Optional parameter that sets the content of the deepest node. Does not change the internal
                             state
        :param create_path_if_not_found: If the the resolved path does not exist, create it by default.
        :return:
        """
        root_found = False
        backup_node_content = ''
        if node_content != '':
            if path_node in self.path_content.keys():
                backup_node_content = self.path_content[path_node]
                self.path_content[path_node] = node_content
            else:
                backup_node_content = ''
                self.path_content[path_node] = node_content

        tmp_node = anytree.find(self.root_node, filter_= lambda x: x.name == path_node)

        if tmp_node is None:
            raise ValueError('Queried path node {0} that is not present in tree.'.format(path_node))

        path_parts = []
        while not root_found:
            if tmp_node.name.endswith('_name'):
                if tmp_node.name not in self.path_content.keys():
                    raise ValueError(
                        'Tried to resolve path but path_node {:s} has no matching content set.'.format(tmp_node.name))
                path_parts.append(self.path_content[tmp_node.name])
            else:
                path_parts.append(tmp_node.name)
            tmp_node = tmp_node.parent
            if tmp_node == None:
                root_found = True

        if backup_node_content != '':
            self.path_content[path_node] = backup_node_content
        # Join list in reverse order to get correctly ordered path
        resolved_path = os.path.join(*path_parts[::-1])
        # deal with overlong pathes on windows
        # if os.name == 'nt':
        #     abspath = os.path.abspath(resolved_path)
        #     resolved_path = '\\\\?\\' + abspath
        # else:
        #     resolved_path = resolved_path
        
        if not os.path.exists(resolved_path) and create_path_if_not_found:
            os.makedirs(resolved_path)
        return resolved_path

    def resolve_filename(self, path_node, filename, node_content=''):
        """
        Similar to resolve_path. Gives a full path to a file that resides in path_node.
        :param path_node:
        :param filename:
        :param node_content:
        :return:
        """
        path = self.resolve_path(path_node, node_content=node_content, create_path_if_not_found=True)
        return os.path.join(path, filename)

    def add_path_node(self, new_node_name, parent_node_name):
        parent_node = anytree.search.find(self.root_node, lambda node: node.name == parent_node_name)
        duplicate_node = anytree.search.find(parent_node, lambda node: node.name == new_node_name)
        if duplicate_node is None:
            new_node = anytree.Node(new_node_name, parent=parent_node)

    def get_path_node_content(self, node_name):
        return self.path_content[node_name]

    def list_files_in_path_node(self, path_node, node_content='', file_type=''):
        path = self.resolve_path(path_node, node_content, create_path_if_not_found=False)
        onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if file_type != '':
            onlyfiles = [onlyfiles for f in onlyfiles if f.endswith(file_type)]

        return onlyfiles




if __name__ == '__main__':
    test_manager = PathManager()
    content = {'base_path_name': 'vpor_data',
               'topic_name': 'machine_learning',
               'db_name': 'ml_database',
               'collection_name': 'collection__2018_01_0__14_55',
               'staging_name': 'sin_cos_staging',
               'comp_name': 'neural_network_learner',
               'eval_name': 'learner_results'}

    test_manager.update_path_content(content)

    print('Example path: {0}'.format(test_manager.resolve_path('eval_name')))
