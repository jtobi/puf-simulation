import numpy as np


class PUFWrapper:
    """
    The PUFWrapper class is responsible for the simulation of the PUF instances of the PUF population. It creates a
    PUF object for each instance and uses it to create challenge-response sets for training, validation and test.
    See the __init function for important parameters
    """

    def __init__(self, puf_class, puf_parameters):
        """

        :param puf_class: The class of the PUF type that should be simulated
        :param puf_parameters: This dict is passed to the puf_class constructor. Furthermore, it has the following items:
            -'enable_noise': bool Should the responses be noisy.
            -'create_feature_vector': bool Decides whether challenges bits ([0,1]) or feature vector bits ([-1, 1])
                                           should be created.
        """
        self.puf_class = puf_class
        self.puf_parameters = puf_parameters
        self.enable_noise = puf_parameters.get('enable_noise', False)
        self.create_feature_vectors = puf_parameters.get('create_feature_vectors', True)
        self.store_all_instances = puf_parameters.get('store_all_instances', True)
        self.current_puf = None

        self.puf_list = []

    def create_new_instance(self, do_not_store=False):
        """
        Create a new PUF instance object. A reference to this PUF will be stored in the
        puf_list unless self.store_all_instances or do_not_store override this.
        :param do_not_store:
        :return:
        """
        self.current_puf = self.puf_class(**self.puf_parameters)
        if self.store_all_instances and not do_not_store:
            self.puf_list.append(self.current_puf)

        return self.current_puf

    def create_input_outputs(self, num_crps, add_reliabilty_to_output=False, num_repetitions_to_evaluate=11,
                             reliability_type=None, disable_noise_for_responses=False, challenges=None):
        """

        :param num_crps:
        :param add_reliabilty_to_output:
        :param num_repetitions_to_evaluate:
        :param reliability_type:
        :param disable_noise: Overwritte noise setting to no noise for -responses-. Reliabiltiy is not affected
        :return:
        """
        if self.current_puf is None:
            raise Exception('No current PUF has been created.')

        if disable_noise_for_responses:
            enable_noise_for_responses = False
        else:
            enable_noise_for_responses = self.enable_noise

        if challenges is None:
            challenges = self.current_puf.create_random_challenges(num_crps,
                                                               create_feature_vectors=self.create_feature_vectors)
        responses = self.current_puf.compute_response(challenges, enable_noise=enable_noise_for_responses,
                                                      input_is_feature_vector=self.create_feature_vectors)
        if add_reliabilty_to_output:
            reliability = self.current_puf.compute_reliability_response(challenges, num_repetitions_to_evaluate,
                                                                        input_is_feature_vector=self.create_feature_vectors,
                                                                        reliability_type=reliability_type)
            output = np.stack([responses, reliability], -1)
        else:
            output = responses
        return challenges, output

    def get_puf_type_name(self):
        if self.current_puf is None:
            raise Exception('No current PUF has been created.')
        return self.current_puf.get_puf_type_name()

    def get_puf_parameter_identifier(self):
        if self.current_puf is None:
            raise Exception('No current PUF has been created.')
        return self.current_puf.get_puf_parameter_identifier()
