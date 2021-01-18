import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import time
from puf_simulation_framework.attack_framework import AttackObject
from .attack_util import PytorchAttackUtil
from puf_simulation_framework.pufs import PufUtil
from .pytorch_attack_wrapper import PytorchIPUFReliability


class PytorchMultiIPUFReliability(PytorchIPUFReliability):

    def create_model(self, **kwargs):
        import_original_x = kwargs.get('import_original_x', False)
        import_original_y = kwargs.get('import_original_y', False)
        model = self.model_class(**self.model_parameters)

        if import_original_x or import_original_y:
            puf_instance = kwargs['puf_instance']
            puf_weights = puf_instance.export_weights()
            if not import_original_x:
                del puf_weights['x_weights'], puf_weights['x_bias']
            if not import_original_y:
                del puf_weights['y_weights'], puf_weights['y_bias']
            model.import_weights(puf_weights, import_from_single_ipuf=True)
        return model

    def compute_multi_correlation(self, data_input, data_output):
        prediction = self.predict(data_input, round_result=False)
        num_multi_pufs = prediction.shape[0]
        correlations = []
        for index in range(num_multi_pufs):
            correlations.append(np.corrcoef(prediction[index], data_output)[0, 1])
        return np.array(correlations)

    def loss_batch(self, model, loss_func, xb, yb, opt=None):
        xb_float = xb.float()
        model_output = model(xb_float)
        model_weights = model.export_tensor_weights()
        # model_weights = self.model_tensor_weights
        loss = loss_func(model_output, yb, model_weights)
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(xb), model_output

    def create_loss_function(self, fit_parameter, orthogonal_weights=None, pivot=0):
        pivot = self.model_parameters['y_pivot']  # TODO: refactor

        def multi_puf_pearson_loss(x, y, model_weights, orthogonal_weights=orthogonal_weights, pivot=pivot):
            x_y_cost_scale = 0.05
            #x_y_cost_scale = 2
            # x_y_cost_scale = 0.0
            # x_y_cost_scale = 5000
            multi_loss_list = []
            for current_multi_puf in range(x.shape[0]):
                cost_sum = PytorchAttackUtil.pearson_loss(x[current_multi_puf, :], y)
                # x_weights = model_weights['x_weights'][current_multi_puf].squeeze()
                # y_weights = model_weights['y_weights'][current_multi_puf].squeeze()
                # pivot = 32
                # num_stages = 64
                # lookup = list(range(0, pivot)) + list(range(pivot + 1, num_stages + 1))
                #
                # cost_sum = torch.abs(torch.dot(x_weights, y_weights[lookup]))*x_y_cost_scale
                # multi_loss_list.append(cost_sum)
                first_half_all_weights = torch.cat([model_weights['x_weights'][current_multi_puf, :, :pivot],
                                                    model_weights['y_weights'][current_multi_puf, :, :pivot]], dim=0)
                second_half_all_weights = torch.cat([model_weights['x_weights'][current_multi_puf, :, pivot:],
                                                     model_weights['y_weights'][current_multi_puf, :, pivot + 1:]],
                                                    dim=0)
                for ref_index in range(second_half_all_weights.shape[0] - 1):
                    for inner_index in range(ref_index + 1, second_half_all_weights.shape[0]):
                        # cost_sum += torch.abs(torch.dot(all_weights[ref_index], all_weights[inner_index]))*x_y_cost_scale
                        cost_sum += torch.abs(PytorchAttackUtil.pearson_loss(first_half_all_weights[ref_index],
                                                                             first_half_all_weights[
                                                                                 inner_index])) * x_y_cost_scale
                        cost_sum += torch.abs(PytorchAttackUtil.pearson_loss(second_half_all_weights[ref_index],
                                                                             second_half_all_weights[
                                                                                 inner_index])) * x_y_cost_scale
                multi_loss_list.append(cost_sum)

            sum_loss = torch.sum(torch.stack(multi_loss_list))
            # if orthogonal_weights is not None:
            #     m_w = model_weights['weights'].squeeze()
            #     o_w = orthogonal_weights['weights']
            #     ortho_scale = x.shape[0]/400
            #
            #     additional_cost = []
            #     for index in range(o_w.shape[0]):
            #         length = torch.norm(m_w) * torch.norm(o_w[index])
            #         additional_cost.append(torch.abs(torch.dot(m_w, o_w[index]))/length)
            #
            #     additional_cost_sum = torch.sum(torch.stack(additional_cost).flatten()) * ortho_scale
            #     cost_sum += additional_cost_sum

            return sum_loss

        return multi_puf_pearson_loss

    def fit_single_reliability(self, training_input, training_output, validation_input, validation_output, **kwargs):
        current_puf = kwargs['puf_instance']
        puf_parameter = current_puf.export_weights()
        orthogonal_weights = kwargs.get('orthogonal_weights', None)

        if self.model is not None:
            del self.model
            self.model = None

        self.model = self.create_model(**kwargs)
        # Get a reference to the model weights
        self.model_tensor_weights = self.model.export_tensor_weights()

        loss_func = self.create_loss_function(self.fit_parameters, orthogonal_weights=orthogonal_weights)
        opt = self.create_optimizer(self.model)
        train_dl, validation_dl = self.create_data_loaders(training_input, training_output,
                                                           validation_input,
                                                           validation_output)
        num_epochs = self.fit_parameters['num_epochs']
        verbose = self.fit_parameters.get('verbose', False)

        start_time = time.time()
        per_epoch_log = []
        for epoch in range(num_epochs):

            self.model.train()
            for train_in, train_out in train_dl:
                self.loss_batch(self.model, loss_func, train_in, train_out, opt=opt)

            self.model.eval()
            val_loss, val_acc = self.eval_dl(self.model, validation_dl, self.loss_batch, loss_func)
            train_loss, train_acc = self.eval_dl(self.model, train_dl, self.loss_batch, loss_func)
            train_correlation = self.compute_multi_correlation(training_input, training_output)
            validation_correlation = self.compute_multi_correlation(validation_input, validation_output)

            # if verbose:
            model_parameter = self.model.export_weights()

            correlation_string, correlation_results = self.create_correlation_string(puf_parameter, model_parameter)
            correlation_string += self.create_within_model_correlation_string(model_parameter)

            # print("Epoch #{:d}/{:d}: Train corr: {:s}, Val corr: {:s}, {:s}".format(
            #     epoch, num_epochs, str(train_correlation), str(validation_correlation), correlation_string))
            print("Epoch #{:d}/{:d}: {:s}".format(
                epoch, num_epochs, correlation_string))

            current_epoch_log = {'validation_loss': val_loss,
                                 'validation_correlation': validation_correlation,
                                 'training_loss': train_loss,
                                 'training_correlation': train_correlation,
                                 'correlation_results': correlation_results
                                 }
            #current_epoch_log.update(correlation_results)
            per_epoch_log.append(current_epoch_log)
        duration = time.time() - start_time

        fit_log = {'per_epoch_log': per_epoch_log,
                   'final_training_correlation': train_correlation,
                   'final_validation_correlation': validation_correlation,
                   'model_weights': model_parameter,
                   'fit_duration': duration,
                   'correlation_results': correlation_results
                   }
        return fit_log

    def create_model_puf_correlations(self, model_weights, puf_weights, model_is_y, puf_is_y, correlation_type,
                                      num_stages, pivot=None):
        """

        :param model_weights:
        :param puf_weights:
        :param model_is_y:
        :param puf_is_y:
        :param correlation_type:
        :return:
        """

        if correlation_type == 'full_correlation':
            if model_is_y != puf_is_y:
                if pivot is None:
                    raise ValueError
                if model_is_y:
                    model_lookup = list(range(0, pivot)) + list(range(pivot + 1, num_stages + 1))
                    # puf is x
                    puf_lookup = list(range(num_stages))
                if puf_is_y:
                    puf_lookup = list(range(0, pivot)) + list(range(pivot + 1, num_stages + 1))
                    # model is x
                    model_lookup = list(range(num_stages))

            else:
                if model_is_y and puf_is_y:
                    model_lookup = list(range(num_stages + 1))
                    puf_lookup = list(range(num_stages + 1))
                else:
                    model_lookup = list(range(num_stages))
                    puf_lookup = list(range(num_stages))
        elif correlation_type == 'first_half_correlation':
            model_lookup = list(range(pivot))
            puf_lookup = list(range(pivot))
        elif correlation_type == 'second_half_correlation':
            if pivot is None:
                raise ValueError
            if model_is_y:
                model_lookup = list(range(pivot + 1, num_stages + 1))
            else:
                model_lookup = list(range(pivot, num_stages))
            if puf_is_y:
                puf_lookup = list(range(pivot + 1, num_stages + 1))
            else:
                puf_lookup = list(range(pivot, num_stages))
        else:
            raise ValueError

        corr_model_to_puf = []
        for current_model_xor in range(model_weights.shape[0]):
            corr_per_puf_xor = []
            for current_puf_xor in range(puf_weights.shape[0]):
                corr = \
                    np.corrcoef(puf_weights[current_puf_xor, puf_lookup],
                                model_weights[current_model_xor, model_lookup])[
                        0, 1]
                corr_per_puf_xor.append(corr)
            corr_model_to_puf.append(corr_per_puf_xor)
        corr_model_to_puf = np.array(corr_model_to_puf)

        best_corr_index = []
        best_corr = []
        for current_model_xor in range(model_weights.shape[0]):
            current_best_corr_index = np.argmax(np.abs(corr_model_to_puf[current_model_xor]))
            current_best_corr = corr_model_to_puf[current_model_xor, current_best_corr_index]
            best_corr_index.append(current_best_corr_index)
            best_corr.append(current_best_corr)

        return corr_model_to_puf, np.array(best_corr_index), np.array(best_corr)

    def create_correlation_string(self, puf_parameter, model_param):
        puf_x_weights = puf_parameter['x_weights']
        puf_y_weights = puf_parameter['y_weights']
        pivot = self.model_parameters['y_pivot']
        num_stages = self.model_parameters['num_stages']
        num_multi_pufs = self.model_parameters['num_multi_pufs']
        model_x_weights = model_param['x_weights']
        model_y_weights = model_param['y_weights']

        num_model_x_xors = model_x_weights.shape[1]
        num_model_y_xors = model_y_weights.shape[1]

        # correlation_type = 'full_correlation'
        correlation_type = 'second_half_correlation'
        correlation_type = 'first_half_correlation'

        correlation_string = ''
        for current_multi_puf in range(num_multi_pufs):
            current_model_x_weights = model_x_weights[current_multi_puf]
            current_model_y_weights = model_y_weights[current_multi_puf]

            corr_model_x_puf_x, best_index_model_x_puf_x, best_corr_model_x_puf_x = self.create_model_puf_correlations(
                current_model_x_weights, puf_x_weights, False, False, correlation_type, num_stages, pivot)

            corr_model_x_puf_y, best_index_model_x_puf_y, best_corr_model_x_puf_y = self.create_model_puf_correlations(
                current_model_x_weights, puf_y_weights, False, puf_is_y=True, correlation_type=correlation_type,
                num_stages=num_stages, pivot=pivot)

            corr_model_y_puf_y, best_index_model_y_puf_y, best_corr_model_y_puf_y = self.create_model_puf_correlations(
                current_model_y_weights, puf_y_weights, model_is_y=True, puf_is_y=True,
                correlation_type=correlation_type,
                num_stages=num_stages, pivot=pivot)
            corr_model_y_puf_x, best_index_model_y_puf_x, best_corr_model_y_puf_x = self.create_model_puf_correlations(
                current_model_y_weights, puf_x_weights, model_is_y=True, puf_is_y=False,
                correlation_type=correlation_type,
                num_stages=num_stages, pivot=pivot)

            correlation_string += 'Multi-PUF: #{:d} | '.format(current_multi_puf)
            for current_model_x_xor in range(num_model_x_xors):
                if abs(best_corr_model_x_puf_x[current_model_x_xor]) > abs(
                        best_corr_model_x_puf_y[current_model_x_xor]):
                    index = best_index_model_x_puf_x[current_model_x_xor]
                    correlation_string += 'Best x({:d})->x({:d}): {:.3f} '.format(current_model_x_xor, index,
                                                                                  best_corr_model_x_puf_x[
                                                                                      current_model_x_xor])
                else:
                    index = best_index_model_x_puf_y[current_model_x_xor]
                    correlation_string += 'Best x({:d})->y({:d}): {:.3f} '.format(current_model_x_xor, index,
                                                                                  best_corr_model_x_puf_y[
                                                                                      current_model_x_xor])

            for current_model_y_xor in range(num_model_y_xors):
                if abs(best_corr_model_y_puf_x[current_model_y_xor]) > abs(
                        best_corr_model_y_puf_y[current_model_y_xor]):
                    index = best_index_model_y_puf_x[current_model_y_xor]
                    correlation_string += 'Best y({:d})->x({:d}): {:.3f} '.format(current_model_y_xor, index,
                                                                                  best_corr_model_y_puf_x[
                                                                                      current_model_y_xor])
                else:
                    index = best_index_model_y_puf_y[current_model_y_xor]
                    correlation_string += 'Best y({:d})->y({:d}): {:.3f} '.format(current_model_y_xor, index,
                                                                                  best_corr_model_y_puf_y[
                                                                                      current_model_y_xor])

            # result_dict = {'corr_puf_x_to_model_x': corr_puf_x_to_model_x,
            #                'corr_puf_y_to_model_y': corr_puf_y_to_model_y,
            #                'corr_puf_x_to_model_y': corr_puf_x_to_model_y,
            #                'corr_puf_y_to_model_x': corr_puf_y_to_model_x}
        result_dict = {}

        return correlation_string, result_dict

    def create_within_model_correlation_string(self, model_param):
        pivot = self.model_parameters['y_pivot']
        num_stages = self.model_parameters['num_stages']
        num_multi_pufs = self.model_parameters['num_multi_pufs']
        model_x_weights = model_param['x_weights']
        model_y_weights = model_param['y_weights']

        correlation_string = ''
        for current_multi_puf in range(num_multi_pufs):
            current_model_x_weights = model_x_weights[current_multi_puf, :, pivot:]
            current_model_y_weights = model_y_weights[current_multi_puf, :, pivot + 1:]

            all_weights = np.concatenate([current_model_x_weights, current_model_y_weights], 0)
            labels = ['x{:d}'.format(tmp) for tmp in range(current_model_x_weights.shape[0])] + ['y{:d}'.format(tmp) for
                                                                                                 tmp in range(
                    current_model_y_weights.shape[0])]
            correlation_string += 'Multi-PUF: #{:d} | '.format(current_multi_puf)
            for ref_index in range(all_weights.shape[0] - 1):
                for inner_index in range(ref_index + 1, all_weights.shape[0]):
                    corr = np.corrcoef(all_weights[ref_index], all_weights[inner_index])[0, 1]
                    correlation_string += 'Intra {:s}->{:s}: {:.3f} '.format(labels[ref_index], labels[inner_index],
                                                                             corr)
        return correlation_string

    def fit(self, training_input, training_output, validation_input, validation_output, **kwargs):
        training_reliability_output = training_output[:, 1]
        validation_reliability_output = validation_output[:, 1]
        training_feature_vectors = PufUtil.challenge_to_feature_vector(training_input)
        validation_feature_vectors = PufUtil.challenge_to_feature_vector(validation_input)

        # for the first stage, fix a (1,1)-IPUF as the model to fit
        # self.model_parameters['output_type'] = 'y_abs_raw_x_abs_raw'
        # self.model_parameters['output_type'] = 'abs_raw'
        # self.model_parameters['num_x_xors'] = 1
        # self.model_parameters['num_y_xors'] = 1
        self.model_parameters['input_is_feature_vector'] = True
        # self.fit_parameters['num_epochs'] = 20

        num_trials = self.fit_parameters['num_first_stage_trials']

        fit_logs = []
        for current_trial in range(num_trials):
            print('----------------------------------------------------')
            print('Trial #{:d}'.format(current_trial))
            tmp = self.fit_single_reliability(training_feature_vectors, training_reliability_output,
                                              validation_feature_vectors,
                                              validation_reliability_output, **kwargs)
            fit_logs.append(tmp)

        return fit_logs

    def predict_raw(self, input):
        if self.model is None:
            raise Exception
        self.model.eval()
        return self.model(torch.Tensor(input)).data.numpy()


class PytorchMultiIdealizedSecondStageIPUFReliability(PytorchMultiIPUFReliability):

    def create_model(self, **kwargs):
        current_puf = kwargs['puf_instance']
        puf_weights = current_puf.export_weights()
        del puf_weights['x_weights'], puf_weights['x_bias']
        model = self.model_class(**self.model_parameters)
        model.import_weights(puf_weights)
        model.freeze_x_y(freeze_x=False, freeze_y=True)
        return model


class PytorchMultiSecondStageIPUFReliability(PytorchMultiIPUFReliability):
    """
    Model gets candidates for the x layer from the first stage and has its x layer frozen. Afterwards the y layer is
    learned stepwise.
    Each step is consists of a number of trials. After each step potential candidates from the trials are gathered and
    promoted to y-puf candidates. These candidates are then added to the constraints pool.
    """

    def fit(self, training_input, training_output, validation_input, validation_output, **kwargs):
        training_reliability_output = training_output[:, 1]
        validation_reliability_output = validation_output[:, 1]
        training_feature_vectors = PufUtil.challenge_to_feature_vector(training_input)
        validation_feature_vectors = PufUtil.challenge_to_feature_vector(validation_input)

        # for the first stage, fix a (1,1)-IPUF as the model to fit
        # self.model_parameters['output_type'] = 'y_abs_raw_x_abs_raw'
        # self.model_parameters['output_type'] = 'abs_raw'
        # self.model_parameters['num_x_xors'] = 1
        # self.model_parameters['num_y_xors'] = 1
        self.model_parameters['input_is_feature_vector'] = True
        # self.fit_parameters['num_epochs'] = 20

        num_trials = self.fit_parameters['num_second_stage_trials']
        num_steps = self.fit_parameters['num_second_stage_steps']
        per_step_num_epochs = self.fit_parameters.get('per_step_num_epoch', None)

        current_puf = kwargs['puf_instance']
        puf_parameter = current_puf.export_weights()
        puf_y = puf_parameter['y_weights']

        y_weight_candidates = []
        y_bias_candidates = []

        per_step_logs = []
        per_step_fit_logs = []
        for current_step in range(num_steps):
            fit_logs = []
            per_model_y_weights = []
            per_model_y_bias = []
            final_validation_correlations = []

            if per_step_num_epochs is not None:
                self.fit_parameters['num_epochs'] = per_step_num_epochs[current_step]

            for current_trial in range(num_trials):
                print('----------------------------------------------------')
                print('Trial #{:d}'.format(current_trial))
                kwargs['orthogonal_weights'] = y_weight_candidates

                tmp = self.fit_single_reliability(training_feature_vectors, training_reliability_output,
                                                  validation_feature_vectors,
                                                  validation_reliability_output, **kwargs)
                fit_logs.append(tmp)
                model_weights = self.model.export_weights()
                per_model_y_weights.append(model_weights['y_weights'].reshape(-1, model_weights['y_weights'].shape[-1]))
                per_model_y_bias.append(model_weights['y_bias'].reshape(-1, model_weights['y_bias'].shape[-1]))
                final_validation_correlations.append(tmp['final_validation_correlation'])

            per_model_y_weights = np.concatenate(per_model_y_weights, axis=0)
            per_model_y_bias = np.concatenate(per_model_y_bias, axis=0)
            final_validation_correlations = np.concatenate(final_validation_correlations, axis=0)

            y_weight_candidates, y_bias_candidates, step_log = self.update_y_candidates(per_model_y_weights,
                                                                                        per_model_y_bias,
                                                                                        puf_y, y_weight_candidates,
                                                                                        y_bias_candidates,
                                                                                        final_validation_correlations)
            per_step_fit_logs.append(fit_logs)
            per_step_logs.append(step_log)

        complete_fit_log = {'per_step_fit_logs': per_step_fit_logs,
                            'per_step_logs': per_step_logs,
                            'y_weight_candidates': y_weight_candidates,
                            'y_bias_candidates': y_bias_candidates}

        return complete_fit_log

    def update_y_candidates(self, per_model_y_weights, per_model_y_bias, puf_y_weights, y_weight_candidates,
                            y_bias_candidates, model_reliability_correlations):
        selection_strategy = self.fit_parameters.get('candidate_weight_selection_strategy', 'largest_component')

        num_stages = per_model_y_weights.shape[-1] - 1
        corr_model_to_puf, best_corr_index, best_corr = self.create_model_puf_correlations(per_model_y_weights,
                                                                                           puf_y_weights, True, True,
                                                                                           correlation_type='full_correlation',
                                                                                           num_stages=num_stages)


        component_results = PytorchAttackUtil.compute_model_components(per_model_y_weights)
        component_members = component_results['component_members']
        component_sizes = component_results['component_sizes']
        largest_component = np.max(component_sizes)
        component_indices = np.argsort(component_sizes)[::-1]

        chosen_component_indices = component_members[component_indices[0]][0]
        chosen_component_reliability_correlations = model_reliability_correlations[chosen_component_indices]
        chosen_component_index = np.argmax(np.abs(chosen_component_reliability_correlations))

        if selection_strategy == 'largest_component':
            chosen_model_index = chosen_component_indices[chosen_component_index]
        elif selection_strategy == 'greedy_largest_correlation':
            chosen_model_index = np.argmax(model_reliability_correlations)
        else:
            raise ValueError

        model_weight_hit_puf = []
        model_weight_hit_correlation = []
        for model_index in range(per_model_y_weights.shape[0]):
            print(
                'Model y #{:d} hit puf y #{:d} with corr: {:.4f}. This model has a validation reliability correlation of {:.4f}'.format(
                    model_index, best_corr_index[model_index],
                    best_corr[model_index], model_reliability_correlations[model_index]))
            model_weight_hit_puf.append(best_corr_index[model_index])
            model_weight_hit_correlation.append(best_corr[model_index])

        print(
            'Largest component has {:d}/{:d} members. Chosen component member has model_index {:d} which hit puf_y {:d} with corr {:.2f} and has validation reliability correlation of {:.4f} (Strategy {:s})'.format(
                largest_component, per_model_y_weights.shape[0], chosen_model_index,
                best_corr_index[chosen_model_index], best_corr[chosen_model_index],
                model_reliability_correlations[chosen_model_index],
                selection_strategy))

        new_y_weight_candidate = per_model_y_weights[chosen_model_index]
        new_y_bias_candidate = per_model_y_bias[chosen_model_index]

        y_bias_candidates.append(new_y_bias_candidate)
        y_weight_candidates.append(new_y_weight_candidate)

        result_log = {'model_hit_puf': model_weight_hit_puf,
                      'model_hit_puf_correlation': model_weight_hit_correlation,
                      'component_sizes': component_sizes,
                      'chosen_model_index': chosen_model_index,
                      'chosen_model_hit_puf': best_corr_index[chosen_model_index],
                      'chosen_model_reliability_correlation': model_reliability_correlations[chosen_model_index],
                      'model_reliability_correlations': model_reliability_correlations}

        return y_weight_candidates, y_bias_candidates, result_log

    def create_loss_function(self, fit_parameters, orthogonal_weights=None, pivot=0):
        pivot = self.model_parameters['y_pivot']  # TODO: refactor

        def multi_puf_pearson_loss(x, y, model_weights, orthogonal_weights=orthogonal_weights, pivot=pivot):
            x_y_cost_scale = 0.1
            ortho_scale = 0.1
            # x_y_cost_scale = 0.0
            # x_y_cost_scale = 5000
            multi_loss_list = []
            for current_multi_puf in range(x.shape[0]):
                cost_sum = PytorchAttackUtil.pearson_loss(x[current_multi_puf, :], y)

                # first_half_all_weights = torch.cat([model_weights['x_weights'][current_multi_puf, :, :pivot],
                #                                     model_weights['y_weights'][current_multi_puf, :, :pivot]], dim=0)
                # second_half_all_weights = torch.cat([model_weights['x_weights'][current_multi_puf, :, pivot:],
                #                                      model_weights['y_weights'][current_multi_puf, :, pivot + 1:]],
                #                                     dim=0)
                # for ref_index in range(second_half_all_weights.shape[0] - 1):
                #     for inner_index in range(ref_index + 1, second_half_all_weights.shape[0]):
                #         # cost_sum += torch.abs(torch.dot(all_weights[ref_index], all_weights[inner_index]))*x_y_cost_scale
                #         cost_sum += torch.abs(PytorchAttackUtil.pearson_loss(first_half_all_weights[ref_index],
                #                                                              first_half_all_weights[
                #                                                                  inner_index])) * x_y_cost_scale
                #         cost_sum += torch.abs(PytorchAttackUtil.pearson_loss(second_half_all_weights[ref_index],
                #                                                              second_half_all_weights[
                #                                                                  inner_index])) * x_y_cost_scale
                if orthogonal_weights:
                    model_y_weights = model_weights['y_weights']

                    for ortho_weight in orthogonal_weights:
                        cost_sum += torch.abs(PytorchAttackUtil.pearson_loss(model_y_weights[current_multi_puf, 0],
                                                                             torch.tensor(ortho_weight)) * ortho_scale) / len(orthogonal_weights)
                multi_loss_list.append(cost_sum)

            sum_loss = torch.sum(torch.stack(multi_loss_list))

            return sum_loss

        return multi_puf_pearson_loss

    def create_correlation_string(self, puf_parameter, model_param):
        puf_x_weights = puf_parameter['x_weights']
        puf_y_weights = puf_parameter['y_weights']
        pivot = self.model_parameters['y_pivot']
        num_stages = self.model_parameters['num_stages']
        num_multi_pufs = self.model_parameters['num_multi_pufs']
        model_x_weights = model_param['x_weights']
        model_y_weights = model_param['y_weights']

        num_model_y_xors = model_y_weights.shape[1]

        correlation_string = ''
        for current_multi_puf in range(num_multi_pufs):
            current_model_y_weights = model_y_weights[current_multi_puf]

            corr_model_y_puf_y, best_index_model_y_puf_y, best_corr_model_y_puf_y = self.create_model_puf_correlations(
                current_model_y_weights, puf_y_weights, model_is_y=True, puf_is_y=True,
                correlation_type='full_correlation',
                num_stages=num_stages, pivot=pivot)
            corr_model_y_puf_x, best_index_model_y_puf_x, best_corr_model_y_puf_x = self.create_model_puf_correlations(
                current_model_y_weights, puf_x_weights, model_is_y=True, puf_is_y=False,
                correlation_type='full_correlation',
                num_stages=num_stages, pivot=pivot)

            correlation_string += 'Multi-PUF: #{:d} | '.format(current_multi_puf)

            for current_model_y_xor in range(num_model_y_xors):
                if abs(best_corr_model_y_puf_x[current_model_y_xor]) > abs(
                        best_corr_model_y_puf_y[current_model_y_xor]):
                    index = best_index_model_y_puf_x[current_model_y_xor]
                    correlation_string += 'Best y({:d})->x({:d}): {:.3f} '.format(current_model_y_xor, index,
                                                                                  best_corr_model_y_puf_x[
                                                                                      current_model_y_xor])
                else:
                    index = best_index_model_y_puf_y[current_model_y_xor]
                    correlation_string += 'Best y({:d})->y({:d}): {:.3f} '.format(current_model_y_xor, index,
                                                                                  best_corr_model_y_puf_y[
                                                                                      current_model_y_xor])

            # result_dict = {'corr_puf_x_to_model_x': corr_puf_x_to_model_x,
            #                'corr_puf_y_to_model_y': corr_puf_y_to_model_y,
            #                'corr_puf_x_to_model_y': corr_puf_x_to_model_y,
            #                'corr_puf_y_to_model_x': corr_puf_y_to_model_x}
        result_dict = {}

        return correlation_string, result_dict


class PytorchMultiXorReliability(PytorchMultiSecondStageIPUFReliability):
    """
    Perform
    Each step is consists of a number of trials. After each step potential candidates from the trials are gathered and
    promoted to y-puf candidates. These candidates are then added to the constraints pool.
    """

    def fit(self, training_input, training_output, validation_input, validation_output, **kwargs):
        training_reliability_output = training_output[:, 1]
        validation_reliability_output = validation_output[:, 1]
        training_feature_vectors = PufUtil.challenge_to_feature_vector(training_input)
        validation_feature_vectors = PufUtil.challenge_to_feature_vector(validation_input)

        # for the first stage, fix a (1,1)-IPUF as the model to fit
        # self.model_parameters['output_type'] = 'y_abs_raw_x_abs_raw'
        # self.model_parameters['output_type'] = 'abs_raw'
        # self.model_parameters['num_x_xors'] = 1
        # self.model_parameters['num_y_xors'] = 1
        # self.fit_parameters['num_epochs'] = 20

        num_trials = self.fit_parameters['num_trials']
        num_steps = self.fit_parameters['num_steps']
        per_step_num_epochs = self.fit_parameters.get('per_step_num_epoch', None)

        current_puf = kwargs['puf_instance']
        puf_parameter = current_puf.export_weights()
        puf_weights = puf_parameter['weights']
        puf_num_xors = current_puf.num_xors

        current_puf = kwargs['puf_instance']
        indiv_puf_responses = current_puf.compute_response(validation_feature_vectors, enable_noise=False,
                                                           input_is_feature_vector=True,
                                                           merge_individual_outputs=False)

        weight_candidates = []
        bias_candidates = []

        per_step_logs = []
        per_step_fit_logs = []
        per_puf_validation_accuracies = np.zeros(puf_num_xors)
        for current_step in range(num_steps):
            print('----------------------------------------------------')
            print('----------------------------------------------------')
            print('Step#{:d}'.format(current_step))
            fit_logs = []
            per_model_weights = []
            per_model_bias = []
            final_validation_correlations = []

            if per_step_num_epochs is not None:
                self.fit_parameters['num_epochs'] = per_step_num_epochs[current_step]
            model_puf_responses = []
            for current_trial in range(num_trials):
                print('----------------------------------------------------')
                print('Trial #{:d}'.format(current_trial))
                kwargs['orthogonal_weights'] = weight_candidates

                tmp = self.fit_single_reliability(training_feature_vectors, training_reliability_output,
                                                  validation_feature_vectors,
                                                  validation_reliability_output, **kwargs)
                fit_logs.append(tmp)
                model_weights = self.model.export_weights()
                per_model_weights.append(model_weights['weights'].reshape(-1, model_weights['weights'].shape[-1]))
                per_model_bias.append(model_weights['bias'].reshape(-1, model_weights['bias'].shape[-1]))
                final_validation_correlations.append(tmp['final_validation_correlation'])

                tmp_responses = self.model.compute_single_puf_responses(validation_feature_vectors)
                model_puf_responses.append(tmp_responses)
                print()
            model_puf_responses = np.concatenate(model_puf_responses, axis=0)

            per_model_weights = np.concatenate(per_model_weights, axis=0)
            per_model_bias = np.concatenate(per_model_bias, axis=0)
            final_validation_correlations = np.concatenate(final_validation_correlations, axis=0)

            weight_candidates, bias_candidates, step_log = self.update_y_candidates(per_model_weights, per_model_bias,
                                                                                    puf_weights, weight_candidates,
                                                                                    bias_candidates,
                                                                                    final_validation_correlations)
            per_step_fit_logs.append(fit_logs)
            per_step_logs.append(step_log)

            per_puf_validation_accuracies[step_log['chosen_model_hit_puf']] = np.mean(indiv_puf_responses[step_log['chosen_model_hit_puf']] == model_puf_responses[step_log['chosen_model_index']])

        complete_fit_log = {'per_step_fit_logs': per_step_fit_logs,
                            'per_step_logs': per_step_logs,
                            'weight_candidates': weight_candidates,
                            'bias_candidates': bias_candidates,
                            'per_puf_validation_accuracies': per_puf_validation_accuracies}

        return complete_fit_log

    def create_correlation_string(self, puf_parameter, model_param):
        puf_weights = puf_parameter['weights']
        num_multi_pufs = self.model_parameters['num_multi_pufs']
        num_stages = self.model_parameters['num_stages']
        model_weights = model_param['weights']

        num_model_xors = model_weights.shape[1]

        multi_corr_model = []
        multi_best_index = []
        multi_best_corr = []
        correlation_string = ''
        for current_multi_puf in range(num_multi_pufs):
            current_model_weights = model_weights[current_multi_puf]

            corr_model, best_index_model, best_corr_model = self.create_model_puf_correlations(
                current_model_weights, puf_weights, model_is_y=False, puf_is_y=False,
                correlation_type='full_correlation',
                num_stages=num_stages, pivot=-1)

            correlation_string += 'Multi-PUF: #{:d} | '.format(current_multi_puf)

            for current_model_xor in range(num_model_xors):
                index = best_index_model[current_model_xor]
                correlation_string += 'Best y({:d})->y({:d}): {:.3f} '.format(current_model_xor, index,
                                                                              best_corr_model[
                                                                                  current_model_xor])
            multi_corr_model.append(corr_model)
            multi_best_index.append(best_index_model)
            multi_best_corr.append(best_corr_model)

        result_dict = {'correlations': np.stack(multi_corr_model, axis=0),
                       'best_corr_index': np.stack(multi_best_index, axis=0),
                       'best_corr': np.stack(multi_best_corr, axis=0)}

        return correlation_string, result_dict

    def create_loss_function(self, fit_parameters, orthogonal_weights=None, pivot=0):

        def multi_puf_pearson_loss(x, y, model_weights, orthogonal_weights=orthogonal_weights, fit_parameters=fit_parameters):
            x_y_cost_scale = 0.1
            ortho_scale = fit_parameters.get('constraint_scale')
            # x_y_cost_scale = 0.0
            # x_y_cost_scale = 5000
            multi_loss_list = []
            for current_multi_puf in range(x.shape[0]):
                cost_sum = PytorchAttackUtil.pearson_loss(x[current_multi_puf, :], y)

                if orthogonal_weights:
                    current_model_weights = model_weights['weights']

                    for ortho_weight in orthogonal_weights:
                        cost_sum += torch.abs(PytorchAttackUtil.pearson_loss(current_model_weights[current_multi_puf, 0],
                                                                             torch.tensor(ortho_weight)) * ortho_scale)
                multi_loss_list.append(cost_sum)

            sum_loss = torch.sum(torch.stack(multi_loss_list))

            return sum_loss

        return multi_puf_pearson_loss

    def create_within_model_correlation_string(self, model_param):
        return ''



class PytorchMultiClassicReliabilityOnIPUF(PytorchMultiXorReliability):
    """
    Learn the IPUF with classic reliability attack, i.e., the model is a single Arbiter PUF

    Each step consists of a number of trials. After each step potential candidates from the trials are gathered and
    promoted to y-puf candidates. These candidates are then added to the constraints pool.
    """

    def fit(self, training_input, training_output, validation_input, validation_output, **kwargs):
        training_reliability_output = training_output[:, 1]
        validation_reliability_output = validation_output[:, 1]
        training_feature_vectors = PufUtil.challenge_to_feature_vector(training_input)
        validation_feature_vectors = PufUtil.challenge_to_feature_vector(validation_input)

        # for the first stage, fix a (1,1)-IPUF as the model to fit
        # self.model_parameters['output_type'] = 'y_abs_raw_x_abs_raw'
        # self.model_parameters['output_type'] = 'abs_raw'
        # self.model_parameters['num_x_xors'] = 1
        # self.model_parameters['num_y_xors'] = 1
        # self.fit_parameters['num_epochs'] = 20

        num_trials = self.fit_parameters['num_trials']
        num_steps = self.fit_parameters['num_steps']
        per_step_num_epochs = self.fit_parameters.get('per_step_num_epoch', None)

        current_puf = kwargs['puf_instance']
        puf_parameter = current_puf.export_weights()
        puf_y_weights = puf_parameter['y_weights']
        puf_x_weights = puf_parameter['x_weights']
        # puf_num_xors = current_puf.num_xors

        current_puf = kwargs['puf_instance']
        # indiv_puf_responses = current_puf.compute_response(validation_feature_vectors, enable_noise=False,
        #                                                    input_is_feature_vector=True,
        #                                                    merge_individual_outputs=False)

        weight_candidates = []
        bias_candidates = []

        per_step_logs = []
        per_step_fit_logs = []
        # per_puf_validation_accuracies = np.zeros(puf_num_xors)
        for current_step in range(num_steps):
            print('----------------------------------------------------')
            print('----------------------------------------------------')
            print('Step#{:d}'.format(current_step))
            fit_logs = []
            per_model_weights = []
            per_model_bias = []
            final_validation_correlations = []

            if per_step_num_epochs is not None:
                self.fit_parameters['num_epochs'] = per_step_num_epochs[current_step]
            model_puf_responses = []
            for current_trial in range(num_trials):
                print('----------------------------------------------------')
                print('Trial #{:d}'.format(current_trial))
                kwargs['orthogonal_weights'] = weight_candidates

                tmp = self.fit_single_reliability(training_feature_vectors, training_reliability_output,
                                                  validation_feature_vectors,
                                                  validation_reliability_output, **kwargs)
                fit_logs.append(tmp)
                model_weights = self.model.export_weights()
                per_model_weights.append(model_weights['weights'].reshape(-1, model_weights['weights'].shape[-1]))
                per_model_bias.append(model_weights['bias'].reshape(-1, model_weights['bias'].shape[-1]))
                final_validation_correlations.append(tmp['final_validation_correlation'])

                tmp_responses = self.model.compute_single_puf_responses(validation_feature_vectors)
                model_puf_responses.append(tmp_responses)
                print()
            model_puf_responses = np.concatenate(model_puf_responses, axis=0)

            per_model_weights = np.concatenate(per_model_weights, axis=0)
            per_model_bias = np.concatenate(per_model_bias, axis=0)
            final_validation_correlations = np.concatenate(final_validation_correlations, axis=0)

            weight_candidates, bias_candidates, step_log = self.update_arbiter_candidates(per_model_weights, per_model_bias, puf_x_weights,
                                                                                    puf_y_weights, weight_candidates,
                                                                                    bias_candidates,
                                                                                    final_validation_correlations)
            per_step_fit_logs.append(fit_logs)
            per_step_logs.append(step_log)

            #per_puf_validation_accuracies[step_log['chosen_model_hit_puf']] = np.mean(indiv_puf_responses[step_log['chosen_model_hit_puf']] == model_puf_responses[step_log['chosen_model_index']])
            per_puf_validation_accuracies = 0

        complete_fit_log = {'per_step_fit_logs': per_step_fit_logs,
                            'per_step_logs': per_step_logs,
                            'weight_candidates': weight_candidates,
                            'bias_candidates': bias_candidates,
                            'per_puf_validation_accuracies': per_puf_validation_accuracies}

        return complete_fit_log

    def create_correlation_string(self, puf_parameter, model_param):
        puf_x_weights = puf_parameter['x_weights']
        puf_y_weights = puf_parameter['y_weights']
        pivot = self.model_parameters['y_pivot']
        num_stages = self.model_parameters['num_stages']
        num_multi_pufs = self.model_parameters['num_multi_pufs']
        model_weights = model_param['weights']

        num_model_xors = model_weights.shape[1]

        # correlation_type = 'full_correlation'
        correlation_type = 'second_half_correlation'
        correlation_type = 'first_half_correlation'

        correlation_string = ''
        for current_multi_puf in range(num_multi_pufs):
            current_model_weights = model_weights[current_multi_puf]

            corr_model_puf_x, best_index_model_puf_x, best_corr_model_puf_x = self.create_model_puf_correlations(
                current_model_weights, puf_x_weights, False, False, correlation_type, num_stages, pivot)

            corr_model_puf_y, best_index_model_puf_y, best_corr_model_puf_y = self.create_model_puf_correlations(
                current_model_weights, puf_y_weights, False, puf_is_y=True, correlation_type=correlation_type,
                num_stages=num_stages, pivot=pivot)

            correlation_string += 'Multi:#{:d}| '.format(current_multi_puf)
            for current_model_xor in range(num_model_xors):
                if abs(best_corr_model_puf_x[current_model_xor]) > abs(
                        best_corr_model_puf_y[current_model_xor]):
                    index = best_index_model_puf_x[current_model_xor]
                    correlation_string += 'Best w({:d})->x({:d}): {:.3f} '.format(current_model_xor, index,
                                                                                  best_corr_model_puf_x[
                                                                                      current_model_xor])
                else:
                    index = best_index_model_puf_y[current_model_xor]
                    correlation_string += 'Best w({:d})->y({:d}): {:.3f} '.format(current_model_xor, index,
                                                                                  best_corr_model_puf_y[
                                                                                      current_model_xor])

            # result_dict = {'corr_puf_x_to_model_x': corr_puf_x_to_model_x,
            #                'corr_puf_y_to_model_y': corr_puf_y_to_model_y,
            #                'corr_puf_x_to_model_y': corr_puf_x_to_model_y,
            #                'corr_puf_y_to_model_x': corr_puf_y_to_model_x}
        result_dict = {}

        return correlation_string, result_dict

    def create_loss_function(self, fit_parameters, orthogonal_weights=None, pivot=0):
        pivot = self.model_parameters['y_pivot']  # TODO: refactor
        def multi_puf_pearson_loss(x, y, model_weights, orthogonal_weights=orthogonal_weights, pivot=pivot):
            x_y_cost_scale = 0.1
            ortho_scale = 0.1
            # x_y_cost_scale = 0.0
            # x_y_cost_scale = 5000
            multi_loss_list = []
            for current_multi_puf in range(x.shape[0]):
                cost_sum = PytorchAttackUtil.pearson_loss(x[current_multi_puf, :], y)

                if orthogonal_weights:
                    first_half_model_weights = model_weights['weights'][current_multi_puf, 0, :pivot]
                    second_half_model_weights = model_weights['weights'][current_multi_puf, 0, pivot:]

                    for ortho_weight in orthogonal_weights:
                        cost_sum += torch.abs(PytorchAttackUtil.pearson_loss(first_half_model_weights, torch.tensor(ortho_weight[:pivot])) * ortho_scale)
                        cost_sum += torch.abs(PytorchAttackUtil.pearson_loss(second_half_model_weights, torch.tensor(ortho_weight[pivot:])) * ortho_scale)

                multi_loss_list.append(cost_sum)

            sum_loss = torch.sum(torch.stack(multi_loss_list))

            return sum_loss

        return multi_puf_pearson_loss

    def create_within_model_correlation_string(self, model_param):
        return ''

    def update_arbiter_candidates(self, per_model_weights, per_model_y_bias, puf_x_weights, puf_y_weights, weight_candidates,
                                  bias_candidates, model_reliability_correlations):
        pivot = self.model_parameters['y_pivot']
        num_stages = per_model_weights.shape[-1]
        component_results = PytorchAttackUtil.compute_model_components(per_model_weights[:, :pivot])
        component_members = component_results['component_members']
        component_sizes = component_results['component_sizes']
        most_component_members = np.max(component_sizes)

        # num_components_with_most_members = np.sum(most_component_members == component_sizes)
        component_indices_with_most_members = np.where(most_component_members == component_sizes)[0]
        # component_with_most_members = component_members[component_indices_with_most_members]

        per_component_best_reliability = []
        for members in component_members:
            per_component_best_reliability.append(np.max(np.abs(model_reliability_correlations[members])))

        per_component_best_reliability = np.array(per_component_best_reliability)

        best_component_index = np.argmax(per_component_best_reliability[component_indices_with_most_members.tolist()])

        # component_indices = np.argsort(component_sizes)[::-1]
        chosen_component_indices = component_members[best_component_index][0]
        chosen_component_size = len(component_members[best_component_index][0])
        chosen_component_reliability_correlations = model_reliability_correlations[chosen_component_indices]
        chosen_component_index = np.argmax(np.abs(chosen_component_reliability_correlations))
        chosen_model_index = chosen_component_indices[chosen_component_index]

        corr_model_to_puf_x, best_corr_index_x, best_corr_x = self.create_model_puf_correlations(per_model_weights,
                                                                                           puf_x_weights, False, False,
                                                                                           correlation_type='full_correlation',
                                                                                           num_stages=num_stages, pivot=pivot)

        corr_model_to_puf_y, best_corr_index_y, best_corr_y = self.create_model_puf_correlations(per_model_weights,
                                                                                           puf_y_weights, False, True,
                                                                                           correlation_type='first_half_correlation',
                                                                                           num_stages=num_stages, pivot=pivot)

        model_weight_hit_puf = []
        model_weight_hit_correlation = []
        for model_index in range(per_model_weights.shape[0]):
            if abs(best_corr_x[model_index]) > abs(best_corr_y[model_index]):
                print(
                'Model #{:d} hit puf x #{:d} with corr: {:.4f}. This model has a validation reliability correlation of {:.4f}'.format(
                        model_index, best_corr_index_x[model_index],
                    best_corr_x[model_index], model_reliability_correlations[model_index]))
                model_weight_hit_puf.append('x({:d})'.format(best_corr_index_x[model_index]))
                model_weight_hit_correlation.append(best_corr_x[model_index])
                if model_index == chosen_model_index:
                    chosen_best_corr = best_corr_x[model_index]
                    chosen_best_corr_index = best_corr_index_x[chosen_model_index]
                    chosen_hit_puf_type = 'x'
            else:
                print(
                    'Model #{:d} hit puf y #{:d} with corr: {:.4f}. This model has a validation reliability correlation of {:.4f}'.format(
                        model_index, best_corr_index_y[model_index],
                        best_corr_y[model_index], model_reliability_correlations[model_index]))
                model_weight_hit_puf.append('y({:d})'.format(best_corr_index_y[model_index]))
                model_weight_hit_correlation.append(best_corr_y[model_index])
                if model_index == chosen_model_index:
                    chosen_best_corr = best_corr_y[model_index]
                    chosen_best_corr_index = best_corr_index_y[chosen_model_index]
                    chosen_hit_puf_type = 'y'

        print(
            'Largest component has {:d}/{:d} members. Chosen component member has model_index {:d} which hit puf {:s} {:d} with corr {:.2f} and has validation reliability correlation of {:.4f}'.format(
                chosen_component_size, per_model_weights.shape[0], chosen_model_index, chosen_hit_puf_type,
                chosen_best_corr_index, chosen_best_corr,
                model_reliability_correlations[chosen_model_index]))

        new_weight_candidate = per_model_weights[chosen_model_index]
        new_bias_candidate = per_model_y_bias[chosen_model_index]

        bias_candidates.append(new_bias_candidate)
        weight_candidates.append(new_weight_candidate)

        result_log = {'model_hit_puf': model_weight_hit_puf,
                      'model_hit_puf_correlation': model_weight_hit_correlation,
                      'component_sizes': component_sizes,
                      'chosen_model_index': chosen_model_index,
                      'chosen_model_hit_puf_type': chosen_hit_puf_type,
                      'chosen_model_hit_puf': chosen_best_corr_index,
                      'chosen_model_reliability_correlation': model_reliability_correlations[chosen_model_index],
                      'model_reliability_correlations': model_reliability_correlations}

        return weight_candidates, bias_candidates, result_log
