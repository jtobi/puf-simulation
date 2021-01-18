import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
from scipy.sparse.csgraph import connected_components


class PytorchAttackUtil:

    @staticmethod
    def pearson_loss(x, y, ):
        """
        This function rewards high linear correlation and punishes negative correlation (even more than zero correlation)
        :param x:
        :param y:
        :return: 
        """
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cost = - vx * vy * torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2))
        pearson_loss = torch.sum(cost)
        return pearson_loss

    @staticmethod
    def pearson_loss_row_wise(x, y):
        """
        This function rewards high linear correlation and punishes negative correlation (even more than zero correlation)
        :param x:
        :param y:
        :return:
        """
        vx = x - torch.mean(x, axis=-1).unsqueeze(-1)
        vy = y - torch.mean(y, axis=-1).unsqueeze(-1)
        cost = - vx * vy * (
                torch.rsqrt(torch.sum(vx ** 2, axis=-1)) * torch.rsqrt(torch.sum(vy ** 2, axis=-1))).unsqueeze(
            axis=-1)
        pearson_loss = torch.sum(cost, axis=-1)
        return pearson_loss

    @staticmethod
    def pearson_loss_rows_with_matrix(x, y):
        """
        x and y are matrices. The pearson loss of each row of x with the complete matrix of y is computed
        :param x:
        :param y:
        :return:
        """
        vx = x - torch.mean(x, axis=-1).unsqueeze(-1)
        vy = y - torch.mean(y, axis=-1).unsqueeze(-1)
        std_x = torch.rsqrt(torch.sum(vx ** 2, axis=-1)).unsqueeze(1)
        std_y = torch.rsqrt(torch.sum(vy ** 2, axis=-1)).unsqueeze(1)
        cost = (vx @ vy.T) * std_x * std_y.T
        pearson_loss = -(cost.flatten())
        return pearson_loss

    @staticmethod
    def pearson_loss_within_matrix(x):
        """
        x is a matrix. The pearson loss is computed between all rows of the matrix
        :param x:
        :param y:
        :return:
        """
        vx = x - torch.mean(x, axis=-1).unsqueeze(-1)
        std_x = torch.rsqrt(torch.sum(vx ** 2, axis=-1)).unsqueeze(1)
        losses = []
        for current_index in range(x.shape[0]-1):
            cost = (vx[current_index] @ vx[current_index+1:].T) * std_x[current_index] * std_x[current_index+1:].T
            losses.append(cost.flatten())
        pearson_loss = -(torch.cat(losses))
        return pearson_loss

    @staticmethod
    def squared_pearson_loss(x, y):
        p_loss = PytorchAttackUtil.pearson_loss(x, y)
        return p_loss * p_loss

    @staticmethod
    def abs_pearson_loss(x, y):
        p_loss = PytorchAttackUtil.pearson_loss(x, y)
        return torch.abs(p_loss)

    @staticmethod
    def compute_model_components(model_weights, corr_threshold=0.95):
        candidates_corr = abs(np.corrcoef(model_weights)) > corr_threshold
        num_connected_components, components = connected_components(candidates_corr)
        component_members = [np.where(components == current_comp) for current_comp in range(num_connected_components)]
        component_sizes = [tmp[0].size for tmp in component_members]

        result = {'candidates_corr': candidates_corr,
                  'num_connected_components': num_connected_components,
                  'components': components,
                  'component_members': component_members,
                  'component_sizes': component_sizes}
        return result


class CorrelationUtil():

    @staticmethod
    def correlate_matrix_vector(matrix, vector):
        """
        Pearson correlation rows of matrix with vector. Returns vector
        :param matrix:
        :param vector:
        :return:
        """
        mean_free_matrix = matrix - np.expand_dims(np.mean(matrix, axis=-1), 1)
        mean_free_vector = np.expand_dims(vector - np.mean(vector, axis=-1), 0)
        p_vec = mean_free_matrix * mean_free_vector * 1 / np.expand_dims(
            np.sqrt(np.sum(mean_free_matrix ** 2, axis=-1)) * np.sqrt(np.sum(mean_free_vector ** 2, axis=-1)), 1)
        pearson_c = np.sum(p_vec, axis=-1)
        return pearson_c

    @staticmethod
    def compute_ipuf_multi_to_model_correlations(puf_weights, model_weights, num_stages=64, pivot=32, half_to_corr='first'):
        """

        :param model_weights:
        :param puf_weights:
        :param model_is_y:
        :param puf_is_y:
        :param correlation_type:
        :return:

        """
        puf_x = puf_weights['x_weights']
        puf_y = puf_weights['y_weights']
        if 'y_pseudo_norm' in model_weights.keys():
            has_y_pseudo_norm = True
            y_pseudo_norm = model_weights['y_pseudo_norm']
        else:
            has_y_pseudo_norm = False
        num_multi_pufs = model_weights['x_weights'].shape[0]
        num_x_pufs = puf_x.shape[0]
        num_y_pufs = puf_y.shape[0]

        #lookup = list(range(0, pivot)) + list(range(pivot + 1, num_stages + 1))
        if half_to_corr == 'first':
            lookup_y = list(range(0,pivot))
            lookup_x = list(range(0,pivot))
        elif half_to_corr == 'second':
            lookup_y = list(range(pivot+1, num_stages+1))
            lookup_x = list(range(pivot, num_stages))
        else:
            raise ValueError

        model_x_to_puf_x = np.zeros((num_multi_pufs, num_x_pufs, num_x_pufs))
        model_y_to_puf_y = np.zeros((num_multi_pufs, num_y_pufs, num_y_pufs))
        model_x_to_puf_y = np.zeros((num_multi_pufs, num_x_pufs, num_y_pufs))
        model_y_to_puf_x = np.zeros((num_multi_pufs, num_y_pufs, num_x_pufs))

        best_correlation_lookup = ['x{:d}'.format(tmp) for tmp in range(num_x_pufs)] + ['y{:d}'.format(tmp) for tmp in
                                                                                        range(num_y_pufs)]

        model_x_best_correlation = []
        model_x_hit = []
        model_y_best_correlation = []
        model_y_hit = []
        for current_multi_puf in range(num_multi_pufs):

            model_x = model_weights['x_weights'][current_multi_puf]
            model_y = model_weights['y_weights'][current_multi_puf]

            if has_y_pseudo_norm:
                model_y[:, :pivot+1] *= y_pseudo_norm[current_multi_puf, 0, :][:, np.newaxis]
                model_y[:, pivot+1:] *= y_pseudo_norm[current_multi_puf, 1, :][:, np.newaxis]

            # Correlate x to x
            for x_index in range(num_x_pufs):
                model_x_to_puf_x[current_multi_puf, x_index, :] = CorrelationUtil.correlate_matrix_vector(puf_x,
                                                                                                          model_x[
                                                                                                              x_index])
            # Correlate y to y
            for y_index in range(num_y_pufs):
                model_y_to_puf_y[current_multi_puf, y_index, :] = CorrelationUtil.correlate_matrix_vector(puf_y[:, lookup_y],
                                                                                                          model_y[
                                                                                                              y_index, lookup_y])
            # Correlate x to y
            for x_index in range(num_x_pufs):
                model_x_to_puf_y[current_multi_puf, x_index, :] = CorrelationUtil.correlate_matrix_vector(
                    puf_y[:, lookup_y],
                    model_x[
                        x_index, lookup_x])
            # Correlate y to x
            for y_index in range(num_y_pufs):
                    model_y_to_puf_x[current_multi_puf, y_index, :] = CorrelationUtil.correlate_matrix_vector(
                        puf_x[:, lookup_x], model_y[y_index, lookup_y])

            x_hit = []
            x_hit_correlation = []
            for x_index in range(num_x_pufs):
                tmp_model_x = np.concatenate([model_x_to_puf_x[current_multi_puf, x_index],
                                              model_x_to_puf_y[current_multi_puf, x_index]], axis=-1)
                best_index = np.argmax(np.abs(tmp_model_x))
                x_hit.append(best_correlation_lookup[best_index])
                x_hit_correlation.append(tmp_model_x[best_index])

            model_x_best_correlation.append(x_hit_correlation)
            model_x_hit.append(x_hit)
            y_hit = []
            y_hit_correlation = []
            for y_index in range(num_y_pufs):
                tmp_model_y = np.concatenate([model_y_to_puf_x[current_multi_puf, y_index],
                                              model_y_to_puf_y[current_multi_puf, y_index]], axis=-1)
                best_index = np.argmax(np.abs(tmp_model_y))
                y_hit.append(best_correlation_lookup[best_index])
                y_hit_correlation.append(tmp_model_y[best_index])

            model_y_best_correlation.append(y_hit_correlation)
            model_y_hit.append(y_hit)

        model_x_best_correlation = np.array(model_x_best_correlation)
        model_x_hit = np.array(model_x_hit)
        model_y_best_correlation = np.array(model_y_best_correlation)
        model_y_hit = np.array(model_y_hit)

        result_dict = {'model_x_to_puf_x': model_x_to_puf_x,
                       'model_y_to_puf_y': model_y_to_puf_y,
                       'model_x_to_puf_y': model_x_to_puf_y,
                       'model_y_to_puf_x': model_y_to_puf_x,
                       'model_x_best_correlation': model_x_best_correlation,
                       'model_y_best_correlation': model_y_best_correlation,
                       'model_x_hit': model_x_hit,
                       'model_y_hit': model_y_hit}

        return result_dict

    @staticmethod
    def compute_ipuf_multi_puf_to_xor_model_correlations(puf_weights, model_weights, num_stages=64, pivot=32,
                                                 half_to_corr='first'):
        """

        :param model_weights:
        :param puf_weights:
        :param model_is_y:
        :param puf_is_y:
        :param correlation_type:
        :return:

        """
        puf_x = puf_weights['x_weights']
        puf_y = puf_weights['y_weights']
        if 'y_pseudo_norm' in model_weights.keys():
            has_y_pseudo_norm = True
            y_pseudo_norm = model_weights['y_pseudo_norm']
        else:
            has_y_pseudo_norm = False
        num_multi_pufs = model_weights['weights'].shape[0]
        num_x_pufs = puf_x.shape[0]
        num_y_pufs = puf_y.shape[0]
        num_model_pufs = model_weights['weights'].shape[1]

        # lookup = list(range(0, pivot)) + list(range(pivot + 1, num_stages + 1))
        if half_to_corr == 'first':
            lookup_y = list(range(0, pivot))
            lookup_x = list(range(0, pivot))
        elif half_to_corr == 'second':
            lookup_y = list(range(pivot + 1, num_stages + 1))
            lookup_x = list(range(pivot, num_stages))
        else:
            raise ValueError

        model_x_to_puf_x = np.zeros((num_multi_pufs, num_model_pufs, num_x_pufs))
        model_y_to_puf_y = np.zeros((num_multi_pufs, num_y_pufs, num_y_pufs))
        model_x_to_puf_y = np.zeros((num_multi_pufs, num_model_pufs, num_y_pufs))
        model_y_to_puf_x = np.zeros((num_multi_pufs, num_y_pufs, num_x_pufs))

        best_correlation_lookup = ['x{:d}'.format(tmp) for tmp in range(num_x_pufs)] + ['y{:d}'.format(tmp) for tmp in
                                                                                        range(num_y_pufs)]

        model_x_best_correlation = []
        model_x_hit = []
        model_y_best_correlation = []
        model_y_hit = []
        for current_multi_puf in range(num_multi_pufs):

            model_w = model_weights['weights'][current_multi_puf]

            # Correlate w to x
            for x_index in range(num_model_pufs):
                model_x_to_puf_x[current_multi_puf, x_index, :] = CorrelationUtil.correlate_matrix_vector(puf_x,
                                                                                                          model_w[
                                                                                                              x_index])
            # Correlate w to y
            for x_index in range(num_model_pufs):
                model_x_to_puf_y[current_multi_puf, x_index, :] = CorrelationUtil.correlate_matrix_vector(
                    puf_y[:, lookup_y],
                    model_w[
                        x_index, lookup_x])


            x_hit = []
            x_hit_correlation = []
            for x_index in range(num_model_pufs):
                tmp_model_x = np.concatenate([model_x_to_puf_x[current_multi_puf, x_index],
                                              model_x_to_puf_y[current_multi_puf, x_index]], axis=-1)
                best_index = np.argmax(np.abs(tmp_model_x))
                x_hit.append(best_correlation_lookup[best_index])
                x_hit_correlation.append(tmp_model_x[best_index])

            model_x_best_correlation.append(x_hit_correlation)
            model_x_hit.append(x_hit)


        model_x_best_correlation = np.array(model_x_best_correlation)
        model_x_hit = np.array(model_x_hit)

        result_dict = {'model_x_to_puf_x': model_x_to_puf_x,
                       'model_x_to_puf_y': model_x_to_puf_y,
                       'model_x_best_correlation': model_x_best_correlation,
                       'model_x_hit': model_x_hit}

        return result_dict
