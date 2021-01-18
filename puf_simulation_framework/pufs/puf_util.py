import numpy as np



class PufUtil:


    @staticmethod
    def challenge_to_feature_vector(challenges):
        """

        :param challenges: [num_challenges, num_challenge_bits]
        :return:
        """

        shifted_challenges = 1 - 2*challenges

        feature_vectors = np.zeros_like(challenges)

        feature_vectors[:, -1] = shifted_challenges[:, -1]
        for offset in list(range(0, challenges.shape[1]-1))[::-1]:
            feature_vectors[:, offset] = feature_vectors[:, offset+1] * shifted_challenges[:, offset]

        return feature_vectors

    @staticmethod
    def generate_random_challenges(num_bits_per_challenge, num_pufs, num_challenges, rand_state=None, keep_singular_dimension=False):

        if rand_state is None:
            rand_state = np.random.random.__self__

        if num_pufs == 1 and not keep_singular_dimension:
            challenge_shape = [num_challenges, num_bits_per_challenge]
        else:
            challenge_shape = [num_pufs, num_challenges, num_bits_per_challenge]

        return rand_state.randint(0, 2, size=challenge_shape, dtype=np.int8)


