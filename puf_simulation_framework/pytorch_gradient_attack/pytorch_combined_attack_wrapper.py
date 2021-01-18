import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import time
from puf_simulation_framework.attack_framework import AttackObject
from .attack_util import PytorchAttackUtil
from puf_simulation_framework.pufs import PufUtil, InterposePuf
from .pytorch_attack_wrapper import PytorchIPUFReliability
from .pytorch_attack_wrapper import MultiPytorchWrapper
from .attack_util import CorrelationUtil
from .puf_models import MultiInterposePufNet, MultiXorArbiterNet
import joblib


class CombinedLossFunctions:
    @staticmethod
    def create_xor_loss_function(fit_parameter):
        def multi_puf_pearson_loss(x, y, model_weights, fit_parameter=fit_parameter):
            """
            y contains the puf response. Dimensions are either [batch_size, 2] with [:,0] the response and [:,1]
            the reliability or dimensions is [batch_size, self.num_multi_pufs+1] with [:,:-1] the individual puf
            responses and [:,-1] the overall reliability. Second case is stricly for second stage of multi stage attack.

            Orthogonal weights allow to impose an additional constraint per multi puf.

            :param x:
            :param y:
            :param model_weights:
            :param fit_parameter:
            :return:
            """
            individual_puf_responses = fit_parameter.get('individual_puf_responses_during_optimization', False)
            orthogonal_weights = fit_parameter.get('orthogonal_weights', False)

            if individual_puf_responses:
                puf_response = y[:, :-1]
                puf_reliability = y[:, -1]
            else:
                puf_response = y[:, 0]
                puf_reliability = y[:, 1]

            model_response = x[0]
            model_reliability = x[1]

            multi_loss_list = []

            reliability_loss_multiplier = fit_parameter.get('reliability_loss_multiplier')
            constraint_loss_multiplier = fit_parameter.get('constraint_loss_multiplier')
            prediction_loss_multiplier = fit_parameter.get('prediction_loss_multiplier')
            orthogonal_loss_multiplier = fit_parameter.get('orthogonal_loss_multiplier')
            for current_multi_puf in range(model_response.shape[0]):
                if individual_puf_responses:
                    current_puf_response = puf_response[:, current_multi_puf]
                else:
                    current_puf_response = puf_response

                if fit_parameter.get('enable_reliability_loss'):
                    # compute reliability loss for all arbiters
                    # for current_arbiter in range(model_reliability.shape[1]):
                    #     #cost_sum += PytorchAttackUtil.pearson_loss(model_reliability[current_multi_puf, current_arbiter, :], puf_reliability)
                    #     multi_loss_list.append(PytorchAttackUtil.pearson_loss(model_reliability[current_multi_puf, current_arbiter, :], puf_reliability) * reliability_loss_multiplier)
                    reliability_loss = (PytorchAttackUtil.pearson_loss_row_wise(
                        model_reliability[current_multi_puf, :, :],
                        puf_reliability.unsqueeze(0))).sum() * reliability_loss_multiplier
                    multi_loss_list.append(reliability_loss)
                if fit_parameter.get('constraint_arbiter_weights_within_multipuf'):
                    # constraint arbiters per multi PUF to not converge to the same PUF
                    current_model_weights = model_weights['weights'][current_multi_puf]
                    # for ref_index in range(current_model_weights.shape[0] - 1):
                    #     # for inner_index in range(ref_index + 1, current_model_weights.shape[0]):
                    #     #     # cost_sum += torch.abs(torch.dot(all_weights[ref_index], all_weights[inner_index]))*constraint_loss_multiplier
                    #     #    multi_loss_list.append(torch.abs(PytorchAttackUtil.pearson_loss(current_model_weights[ref_index], current_model_weights[inner_index])) * constraint_loss_multiplier)
                    #     constraint_loss = torch.abs(
                    #         PytorchAttackUtil.pearson_loss_row_wise(current_model_weights[ref_index].unsqueeze(0),
                    #                                                 current_model_weights[
                    #                                                 ref_index + 1:])).sum() * constraint_loss_multiplier
                    #     multi_loss_list.append(constraint_loss)
                    constraint_loss = torch.abs(PytorchAttackUtil.pearson_loss_within_matrix(
                        current_model_weights) * constraint_loss_multiplier)
                    multi_loss_list.append(torch.sum(constraint_loss))

                if fit_parameter.get('constraining_orthogonal_weights', False) and orthogonal_weights is not None:
                    current_model_weights = model_weights['weights'][current_multi_puf]
                    # for ref_index in range(current_model_weights.shape[0]):
                    #     constraint_loss = torch.abs(
                    #         PytorchAttackUtil.pearson_loss_row_wise(current_model_weights[ref_index].unsqueeze(0),
                    #                                                 orthogonal_weights[
                    #                                                     current_multi_puf])).sum() * orthogonal_loss_multiplier
                    #     multi_loss_list.append(constraint_loss)

                    ortho_constraint_loss = torch.abs(
                        PytorchAttackUtil.pearson_loss_rows_with_matrix(current_model_weights,
                                                                        orthogonal_weights[
                                                                            current_multi_puf])) * orthogonal_loss_multiplier
                    multi_loss_list.append(torch.sum(ortho_constraint_loss).squeeze())

                if fit_parameter.get('enable_prediction_loss'):
                    multi_loss_list.append(F.binary_cross_entropy(model_response[current_multi_puf],
                                                                  current_puf_response) * prediction_loss_multiplier)

            return torch.sum(torch.stack(multi_loss_list))

        return multi_puf_pearson_loss

    @staticmethod
    def create_ipuf_loss_function(fit_parameter):

        def multi_puf_pearson_loss(x, y, model_weights, fit_parameter=fit_parameter):

            puf_response = y[:, 0]
            puf_reliability = y[:, 1]

            model_response = x[0]
            model_x_reliability = x[1]
            inverted_model_x_reliability = x[2]
            model_y_reliability = x[3]
            inverted_model_y_reliability = x[4]

            # cost_sum = torch.tensor(()).new_zeros(1, dtype=torch.float32, requires_grad=True)
            multi_loss_list = []

            y_reliability_loss_multiplier = fit_parameter['y_reliability_loss_multiplier']
            x_reliability_loss_multiplier = fit_parameter['x_reliability_loss_multiplier']
            constraint_loss_multiplier = fit_parameter['constraint_loss_multiplier']
            x_to_y_constraint_loss_multiplier = fit_parameter.get('x_to_y_constraint_loss_multiplier')
            prediction_loss_multiplier = fit_parameter['prediction_loss_multiplier']
            y_pivot = fit_parameter['y_pivot']
            for current_multi_puf in range(model_response.shape[0]):

                if fit_parameter['enable_reliability_loss']:
                    x_reliability_loss = (PytorchAttackUtil.pearson_loss_row_wise(
                        model_x_reliability[current_multi_puf, :, :],
                        puf_reliability.unsqueeze(0))).sum() * x_reliability_loss_multiplier
                    inverted_x_reliability_loss = (PytorchAttackUtil.pearson_loss_row_wise(
                        inverted_model_x_reliability[current_multi_puf, :, :],
                        puf_reliability.unsqueeze(0))).sum() * x_reliability_loss_multiplier
                    inverted_x_reliability_loss = -torch.abs(inverted_x_reliability_loss)  # punish high inverted X loss
                    y_reliability_loss = (PytorchAttackUtil.pearson_loss_row_wise(
                        model_y_reliability[current_multi_puf, :, :],
                        puf_reliability.unsqueeze(0))).sum() * y_reliability_loss_multiplier
                    inverted_y_reliability_loss = (PytorchAttackUtil.pearson_loss_row_wise(
                        inverted_model_y_reliability[current_multi_puf, :, :],
                        puf_reliability.unsqueeze(0))).sum() * y_reliability_loss_multiplier
                    multi_loss_list += [x_reliability_loss, inverted_x_reliability_loss, y_reliability_loss,
                                        inverted_y_reliability_loss]

                if fit_parameter['constraint_arbiter_weights_within_multipuf']:
                    # # constraint arbiters per multi PUF to not converge to the same PUF
                    current_model_x_weights = model_weights['x_weights'][current_multi_puf]
                    current_model_y_weights = model_weights['y_weights'][current_multi_puf]
                    # for ref_index in range(current_model_x_weights.shape[0] - 1):
                    #     x_constraint_loss = torch.abs(
                    #         PytorchAttackUtil.pearson_loss_row_wise(current_model_x_weights[ref_index].unsqueeze(0),
                    #                                                 current_model_x_weights[
                    #                                                 ref_index + 1:])).sum() * constraint_loss_multiplier
                    if current_model_x_weights.shape[0] > 1:
                        x_constraint_loss = torch.abs(PytorchAttackUtil.pearson_loss_within_matrix(
                            current_model_x_weights)) * constraint_loss_multiplier
                        multi_loss_list.append(torch.sum(x_constraint_loss))
                    # for ref_index in range(current_model_y_weights.shape[0] - 1):
                    #     y_constraint_loss = torch.abs(
                    #         PytorchAttackUtil.pearson_loss_row_wise(current_model_y_weights[ref_index].unsqueeze(0),
                    #                                                 current_model_y_weights[
                    #                                                 ref_index + 1:])).sum() * constraint_loss_multiplier
                    if current_model_y_weights.shape[0] > 1:
                        y_constraint_loss = torch.abs(PytorchAttackUtil.pearson_loss_within_matrix(
                            current_model_y_weights)) * constraint_loss_multiplier
                        multi_loss_list.append(torch.sum(y_constraint_loss))
                    # for ref_index in range(current_model_x_weights.shape[0]):
                    #     x_y_constraint_loss_first_half = torch.abs(
                    #         PytorchAttackUtil.pearson_loss_row_wise(
                    #             current_model_x_weights[ref_index, :y_pivot].unsqueeze(0),
                    #             current_model_y_weights[
                    #             :, :y_pivot])).sum() * constraint_loss_multiplier
                    #     x_y_constraint_loss_second_half = torch.abs(
                    #         PytorchAttackUtil.pearson_loss_row_wise(
                    #             current_model_x_weights[ref_index, y_pivot:].unsqueeze(0),
                    #             current_model_y_weights[
                    #             :, y_pivot + 1:])).sum() * constraint_loss_multiplier
                if fit_parameter.get('enable_x_to_y_constraint_loss'):
                    x_y_constraint_loss_first_half = PytorchAttackUtil.pearson_loss_rows_with_matrix(
                        current_model_x_weights[:, :y_pivot],
                        current_model_y_weights[:, :y_pivot])
                    x_y_constraint_loss_first_half = torch.sum(
                        torch.abs(x_y_constraint_loss_first_half)) * x_to_y_constraint_loss_multiplier
                    x_y_constraint_loss_second_half = PytorchAttackUtil.pearson_loss_rows_with_matrix(
                        current_model_x_weights[:, y_pivot:],
                        current_model_y_weights[:, y_pivot + 1:])
                    x_y_constraint_loss_second_half = torch.sum(
                        torch.abs(x_y_constraint_loss_second_half)) * x_to_y_constraint_loss_multiplier
                    multi_loss_list.append(x_y_constraint_loss_first_half)
                    multi_loss_list.append(x_y_constraint_loss_second_half)
                if fit_parameter.get('enable_prediction_loss'):
                    multi_loss_list.append(F.binary_cross_entropy(model_response[current_multi_puf],
                                                                  puf_response) * prediction_loss_multiplier)

            return torch.sum(torch.stack(multi_loss_list))

        return multi_puf_pearson_loss


class CombinedAttackXor(MultiPytorchWrapper):
    """
    Model gets both reliability values and reponse bit for each challenge
    """

    def __init__(self, model_class, model_parameters, optim_parameters, fit_parameters):
        super().__init__(model_class, model_parameters, optim_parameters, fit_parameters)
        self.best_model = 0

    def fit(self, training_input, training_output, validation_input, validation_output, **kwargs):

        num_trials = self.fit_parameters['num_trials']
        kwargs['fit_parameters'] = self.fit_parameters
        kwargs['model_parameters'] = self.model_parameters

        fit_logs = []
        for current_trial in range(num_trials):
            print('----------------------------------------------------')
            print('Trial #{:d}'.format(current_trial))
            tmp = self.fit_single_reliability(training_input, training_output,
                                              validation_input,
                                              validation_output, **kwargs)
            fit_logs.append(tmp)

        return fit_logs

    def debug_acc(self, model, validation_dl, loss_func, fit_parameters, model_parameters):
        tmp_model = MultiInterposePufNet(**model_parameters)
        tmp_model.import_weights(model.export_weights(), False)
        a = 1
        # for batch_in, batch_out in validation_dl:
        #     loss, num_items, model_output = self.loss_batch(model, loss_func, batch_in, batch_out)
        #     loss, num_items, model_output = self.loss_batch(tmp_model, loss_func, batch_in, batch_out)
        val_loss, val_acc = self.eval_dl(tmp_model, validation_dl, self.loss_batch, loss_func, fit_parameters)
        acc_string = 'AARRGH: '
        acc_string += ''.join(
            ['| #{:d} val acc {:.2f}'.format(tmp, val_acc[tmp]) for tmp in range(len(val_acc))])
        print(acc_string)

    def fit_single_reliability(self, training_input, training_output, validation_input, validation_output, **kwargs):
        current_puf = kwargs['puf_instance']
        puf_weights = current_puf.export_weights()
        fit_parameters = kwargs['fit_parameters']
        # model_parameters = kwargs['model_parameters']

        if self.model is not None:
            del self.model
            self.model = None

        self.model = self.create_model(**kwargs)
        # Get a reference to the model weights
        self.model_tensor_weights = self.model.export_tensor_weights()

        loss_func = self.create_loss_function(fit_parameters)
        opt = self.create_optimizer(self.model)
        train_dl, validation_dl = self.create_data_loaders(training_input, training_output,
                                                           validation_input,
                                                           validation_output,
                                                           fit_parameters)
        num_epochs = fit_parameters['num_epochs']

        start_time = time.time()
        per_epoch_log = []
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.per_epoch_hook(epoch)
            # self.debug_acc(self.model, validation_dl, loss_func, fit_parameters, model_parameters)
            self.model.train()
            for train_in, train_out in train_dl:
                self.loss_batch(self.model, loss_func, train_in, train_out, opt=opt)

            self.model.eval()
            val_loss, val_acc = self.eval_dl(self.model, validation_dl, self.loss_batch, loss_func, fit_parameters)
            train_loss, train_acc = self.eval_dl(self.model, train_dl, self.loss_batch, loss_func, fit_parameters)
            # train_correlation = self.compute_multi_correlation(training_input, training_output)
            # validation_correlation = self.compute_multi_correlation(validation_input, validation_output)
            # self.debug_acc(self.model, validation_dl, loss_func, fit_parameters, model_parameters)

            # if verbose:
            model_weights = self.model.export_weights()

            correlation_string, correlation_results = self.create_correlation_string(puf_weights, model_weights,
                                                                                     **kwargs)
            # correlation_string += self.create_within_model_correlation_string(model_parameter)
            acc_string = ''.join(
                ['| #{:d} train acc {:.2f}'.format(tmp, train_acc[tmp]) for tmp in range(len(train_acc))])
            acc_string += '\n'
            acc_string += ''.join(
                ['| #{:d} val acc {:.2f}'.format(tmp, val_acc[tmp]) for tmp in range(len(val_acc))])
            if fit_parameters.get('print_loss'):
                acc_string += '\n val loss {:.2f}'.format(val_loss)
            correlation_string = acc_string + correlation_string

            epoch_duration = time.time() - epoch_start_time

            # print("Epoch #{:d}/{:d}: Train corr: {:s}, Val corr: {:s}, {:s}".format(
            #     epoch, num_epochs, str(train_correlation), str(validation_correlation), correlation_string))
            print("Epoch #{:d}/{:d} ({:.2f}s): {:s}".format(
                epoch, num_epochs, epoch_duration, correlation_string))

            current_epoch_log = {'validation_loss': val_loss,
                                 'validation_accuracy': val_acc,
                                 # 'validation_correlation': validation_correlation,
                                 'training_loss': train_loss,
                                 'training_accuracy': train_acc,
                                 # 'training_correlation': train_correlation,
                                 'correlation_results': correlation_results

                                 }
            # current_epoch_log.update(correlation_results)
            per_epoch_log.append(current_epoch_log)
        duration = time.time() - start_time

        self.best_model = np.argmax(val_acc)

        fit_log = {'per_epoch_log': per_epoch_log,
                   # 'final_training_correlation': train_correlation,
                   # 'final_validation_correlation': validation_correlation,
                   'model_weights': model_weights,
                   'puf_weights': puf_weights,
                   'fit_duration': duration,
                   'final_validation_accuracy': val_acc,
                   'final_training_accuracy': train_acc,
                   # 'correlation_results': correlation_results
                   }
        return fit_log

    def create_model(self, **kwargs):
        model = self.model_class(**self.model_parameters)
        return model

    def loss_batch(self, model, loss_func, xb, yb, opt=None):
        model_output = model(xb)
        model_weights = model.export_tensor_weights()
        loss = loss_func(model_output, yb, model_weights)
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(xb), model_output

    def per_epoch_hook(self, current_epoch):
        pass

    def create_loss_function(self, fit_parameter):
        return CombinedLossFunctions.create_xor_loss_function(fit_parameter)

    def eval_dl(self, model, dataloader, loss_batch, loss_func, fit_parameters):
        num_multi_pufs = model.num_multi_pufs
        loss_cum = 0
        total_num_samples = 0
        accuracies_cum = np.zeros((num_multi_pufs,))
        for batch_in, batch_out in dataloader:
            loss, num_items, model_output = loss_batch(model, loss_func, batch_in, batch_out)
            prediction = model_output[0]
            puf_response = batch_out[:, 0].type(torch.BoolTensor).squeeze()

            loss_cum += loss
            total_num_samples += num_items
            prediction = prediction > 0.5
            current_accuracies = np.array(
                [int(((prediction[multi_index]) == puf_response).sum()) for multi_index in range(num_multi_pufs)])
            accuracies_cum += current_accuracies

        return loss_cum / total_num_samples, accuracies_cum / total_num_samples

    def create_correlation_string(self, puf_weights, model_weights, **kwargs):
        return '', None

    def predict(self, input, round_result=True):
        if self.model is None:
            raise Exception
        self.model.eval()
        # select prediction from model output, disregard reliability output
        model_response = self.model(torch.Tensor(input))
        # if responses and reliability are available, only use responses
        if isinstance(model_response, tuple):
            model_response = model_response[0]
        result = model_response.data.numpy()[self.best_model]  # ToDo: Make sure that best_model always exists
        if round_result:
            result = np.round(result)
        return result


class CombinedAttackIPUF(CombinedAttackXor):

    def create_loss_function(self, fit_parameter):
        return CombinedLossFunctions.create_ipuf_loss_function(fit_parameter)

    def create_correlation_string(self, puf_weights, model_weights, **kwargs):
        print_complete_correlation = kwargs.get('print_complete_correlation', False)
        return self.create_correlation_string_ipuf(puf_weights, model_weights, kwargs['model_parameters'],
                                                   plot_y_correlation=True,
                                                   print_both_halves=print_complete_correlation)

    def create_correlation_string_ipuf(self, puf_weights, model_weights, model_parameters, plot_y_correlation=True,
                                       print_both_halves=False):
        num_stages = model_parameters['num_stages']
        pivot = model_parameters['y_pivot']
        num_multi_pufs = model_parameters['num_multi_pufs']
        num_x_pufs = model_parameters['num_x_xors']
        num_y_pufs = model_parameters['num_y_xors']
        first_half_model_to_puf_correlation = CorrelationUtil.compute_ipuf_multi_to_model_correlations(puf_weights,
                                                                                                       model_weights,
                                                                                                       num_stages,
                                                                                                       pivot, 'first')
        second_half_model_to_puf_correlation = CorrelationUtil.compute_ipuf_multi_to_model_correlations(puf_weights,
                                                                                                        model_weights,
                                                                                                        num_stages,
                                                                                                        pivot, 'second')

        result_string = ''
        for multi_puf in range(num_multi_pufs):
            result_string += '\n#{:d}'.format(multi_puf)
            for x_puf in range(num_x_pufs):
                result_string += '| x{:d} hit {:s} ({: .2f})'.format(x_puf,
                                                                     first_half_model_to_puf_correlation['model_x_hit'][
                                                                         multi_puf, x_puf],
                                                                     first_half_model_to_puf_correlation[
                                                                         'model_x_best_correlation'][multi_puf,
                                                                                                     x_puf])
            if plot_y_correlation:
                for y_puf in range(num_y_pufs):
                    result_string += '| y{:d} hit {:s} ({: .2f})'.format(y_puf,
                                                                         first_half_model_to_puf_correlation[
                                                                             'model_y_hit'][
                                                                             multi_puf, y_puf],
                                                                         first_half_model_to_puf_correlation[
                                                                             'model_y_best_correlation'][multi_puf,
                                                                                                         y_puf])
                    if print_both_halves:
                        result_string += ' and {:s} ({: .2f}) '.format(
                            second_half_model_to_puf_correlation[
                                'model_y_hit'][
                                multi_puf, y_puf],
                            second_half_model_to_puf_correlation[
                                'model_y_best_correlation'][multi_puf,
                                                            y_puf])

        return result_string, first_half_model_to_puf_correlation

    def per_epoch_hook(self, current_epoch):
        if self.fit_parameters.get('reset_x_layer', False):

            if current_epoch == self.fit_parameters['reset_x_at_epoch']:
                print('Resetting x arbiter layer weights')
                self.model.x_arbiter_layer.reset_parameters()

    def create_model(self, **kwargs):
        import_original_x = kwargs['fit_parameters'].get('import_original_x', False)
        import_original_y = kwargs['fit_parameters'].get('import_original_y', False)
        invert_one_y_arbiter = kwargs['fit_parameters'].get('invert_one_y_arbiter', False)
        invert_one_x_arbiter = kwargs['fit_parameters'].get('invert_one_x_arbiter', False)
        model = self.model_class(**self.model_parameters)

        if import_original_x or import_original_y:
            puf_instance = kwargs['puf_instance']
            puf_weights = puf_instance.export_weights()
            if not import_original_x:
                del puf_weights['x_weights'], puf_weights['x_bias']
            if not import_original_y:
                del puf_weights['y_weights'], puf_weights['y_bias']
            model.import_weights(puf_weights, import_from_single_ipuf=True, invert_one_y_arbiter=invert_one_y_arbiter,
                                 invert_one_x_arbiter=invert_one_x_arbiter)
        return model


class CombinedMultiStageAttackIPUF(CombinedAttackIPUF):

    def __init__(self, model_class, model_parameters, optim_parameters, fit_parameters):
        super().__init__(model_class, model_parameters, optim_parameters, fit_parameters)

        self.current_model_parameters = None
        self.initial_model_weights = None
        self.reference_ipuf_model_parameters = None  # hack for printing model correlations
        self.current_loss_function_type = None

    def create_loss_function(self, fit_parameter):
        if self.current_loss_function_type == 'xor':
            return CombinedLossFunctions.create_xor_loss_function(fit_parameter)
        elif self.current_loss_function_type == 'ipuf':
            return CombinedLossFunctions.create_ipuf_loss_function(fit_parameter)
        else:
            raise ValueError

    def create_model(self, **kwargs):
        model = self.model_class(**self.current_model_parameters)
        if self.initial_model_weights is not None:
            # TODO: This should be more generic
            model.import_weights(self.initial_model_weights, import_from_single_ipuf=False)
        return model

    def fit(self, training_input, training_output, validation_input, validation_output, **kwargs):
        complete_logs = {}
        perform_first_stage = kwargs.get('perform_first_stage', True)
        perform_second_stage = kwargs.get('perform_second_stage', True)
        perform_third_stage = kwargs.get('perform_third_stage', True)

        first_stage_results = {}
        second_stage_results = {}
        third_stage_results = {}
        if perform_first_stage:
            # for the first stage, fix a (1,1)-IPUF as the model to fit
            if kwargs.get('load_first_stage', False):
                print('Loading first stage results')
                stage_results = self.load_stage(kwargs['first_stage_tag'])
                first_stage_results = stage_results['stage_results']
                training_input, training_output = stage_results['training_input'], stage_results['training_output']
                validation_input, validation_output = stage_results['validation_input'], stage_results[
                    'validation_output']
                loaded_kwargs = stage_results['kwargs']
                kwargs['puf_instance'] = loaded_kwargs['puf_instance']
            else:
                first_stage_results = self.perform_first_stage(training_input, training_output, validation_input,
                                                               validation_output, **kwargs)
                if kwargs.get('store_first_stage', False):
                    print('Storing first stage results')
                    self.store_stage(training_input, training_output, validation_input, validation_output,
                                     first_stage_results,
                                     kwargs['first_stage_tag'], store_in_out=True, store_kwargs=True, **kwargs)
            complete_logs['first_stage'] = first_stage_results

        if perform_second_stage:
            if kwargs.get('load_second_stage', False):
                print('Loading second stage results')
                stage_results = self.load_stage(kwargs['second_stage_tag'])
                second_stage_results = stage_results['stage_results']
            else:
                second_stage_results = self.perform_second_stage(training_input, training_output, validation_input,
                                                                 validation_output, first_stage_results, **kwargs)
                if kwargs.get('store_second_stage', False):
                    print('Storing second stage results')
                    self.store_stage(training_input, training_output, validation_input, validation_output,
                                     second_stage_results,
                                     kwargs['second_stage_tag'], store_in_out=False, store_kwargs=False)
            complete_logs['second_stage'] = second_stage_results

        if perform_third_stage:
            third_stage_results = self.perform_third_stage(training_input, training_output, validation_input,
                                                           validation_output, first_stage_results, second_stage_results,
                                                           **kwargs)
            complete_logs['third_stage'] = third_stage_results

        return complete_logs

    def perform_first_stage(self, training_input, training_output, validation_input, validation_output, **kwargs):
        # First stage, find y-pufs
        print('----------------------------------------------')
        print('First Stage')
        print('----------------------------------------------')
        self.current_model_parameters = self.model_parameters['first_stage']
        self.reference_ipuf_model_parameters = self.model_parameters['first_stage']
        kwargs['fit_parameters'] = self.fit_parameters['first_stage']
        kwargs['model_parameters'] = self.current_model_parameters
        self.current_loss_function_type = 'ipuf'
        self.model_class = MultiInterposePufNet
        model = self.create_model(**self.current_model_parameters)
        first_stage_model_weights = model.export_weights()
        model.import_weights(first_stage_model_weights, False)
        fit_log_1 = self.fit_single_reliability(training_input, training_output,
                                                validation_input,
                                                validation_output, **kwargs)
        first_stage_weights = self.model.export_weights()
        puf_weights = kwargs['puf_instance'].export_weights()
        puf_parameters = self.fit_parameters['first_stage']['puf_parameter']
        training_x_responses, training_x_accuracy = self.compute_x_hypo_responses(self.model, training_input,
                                                                                  training_output[:, 0],
                                                                                  puf_parameters, puf_weights)
        validation_x_responses, validation_x_accuracy = self.compute_x_hypo_responses(self.model, validation_input,
                                                                                      validation_output[:, 0],
                                                                                      puf_parameters, puf_weights)

        first_stage_result = {'training_x_responses': training_x_responses,
                              'validation_x_responses': validation_x_responses,
                              'first_stage_weights': first_stage_weights,
                              'first_stage_fit_log': fit_log_1,
                              'training_x_accuracy': training_x_accuracy,
                              'validation_x_accuracy': validation_x_accuracy}

        return first_stage_result

    def perform_second_stage(self, training_input, training_output, validation_input, validation_output,
                             first_stage_result, **kwargs):

        training_x_responses = first_stage_result['training_x_responses']
        validation_x_responses = first_stage_result['validation_x_responses']
        training_x_accuracy = first_stage_result['training_x_accuracy']
        validation_x_accuracy = first_stage_result['validation_x_accuracy']
        first_stage_model_weights = first_stage_result['first_stage_weights']
        self.reference_ipuf_model_parameters = self.model_parameters['first_stage']  # hack

        # Second stage, find x-pufs
        print('----------------------------------------------')
        print('Second Stage')
        print('----------------------------------------------')

        print('Training x accuracy {:s}'.format(str(training_x_accuracy)))
        print('Validation x accuracy {:s}'.format(str(validation_x_accuracy)))

        pivot = kwargs['puf_instance'].y_pivot
        num_stages = kwargs['puf_instance'].num_stages
        lookup = list(range(0, pivot)) + list(range(pivot + 1, num_stages + 1))
        self.current_model_parameters = self.model_parameters['second_stage']
        orthogonal_y_weights = first_stage_model_weights['y_weights'][:, :, lookup]
        inverted_orthogonal_y_weights = orthogonal_y_weights.copy()
        inverted_orthogonal_y_weights[:, :, :pivot] *= -1
        orthogonal_y_weights = np.concatenate([orthogonal_y_weights, inverted_orthogonal_y_weights], axis=1)
        self.fit_parameters['second_stage']['orthogonal_weights'] = torch.tensor(orthogonal_y_weights)
        kwargs['fit_parameters'] = self.fit_parameters['second_stage']
        kwargs['model_parameters'] = self.current_model_parameters
        self.current_loss_function_type = 'xor'
        self.model_class = MultiXorArbiterNet
        # original_reference_puf_instance = kwargs['puf_instance']
        # The x reponses need to be inverted here? PUF model vs gradient model
        second_stage_training_output = np.concatenate(
            [self.invert_binary_vector(training_x_responses), training_output[:, 1:2]], axis=-1)
        second_stage_validation_output = np.concatenate(
            [self.invert_binary_vector(validation_x_responses), validation_output[:, 1:2]], axis=-1)
        fit_log_2 = self.fit_single_reliability(training_input, second_stage_training_output,
                                                validation_input,
                                                second_stage_validation_output, **kwargs)
        second_stage_weights = self.model.export_weights()
        second_stage_results = {'second_stage_weights': second_stage_weights}
        return second_stage_results

    def perform_third_stage(self, training_input, training_output, validation_input, validation_output,
                            first_stage_result, second_stage_result, **kwargs):
        first_stage_weights = first_stage_result['first_stage_weights']
        second_stage_weights = second_stage_result['second_stage_weights']
        # Third stage, take y-pufs from first stage and x-pufs from second stage
        print('----------------------------------------------')
        print('Third Stage')
        print('----------------------------------------------')
        self.current_model_parameters = self.model_parameters['third_stage']

        if self.fit_parameters['third_stage'].get('try_double_second_stage_models', False):
            stage_three_initial_weights = {'y_weights': np.concatenate([first_stage_weights['y_weights']] * 2, axis=0),
                                           'y_bias': np.concatenate([first_stage_weights['y_bias']] * 2, axis=0),
                                           'x_weights': np.concatenate([second_stage_weights['weights'], ] * 2, axis=0),
                                           'x_bias': np.concatenate([second_stage_weights['bias']] * 2, axis=0)}
            num_multi_pufs = self.current_model_parameters['num_multi_pufs']
            for current_multi_index in range(num_multi_pufs, num_multi_pufs * 2):
                stage_three_initial_weights['x_weights'][current_multi_index, 0] *= -1
                stage_three_initial_weights['x_bias'][current_multi_index, 0] *= -1
            self.initial_model_weights = stage_three_initial_weights
            self.current_model_parameters['num_multi_pufs'] *= 2
        else:
            stage_three_initial_weights = {'y_weights': first_stage_weights['y_weights'],
                                           'y_bias': first_stage_weights['y_bias'],
                                           'x_weights': second_stage_weights['weights'],
                                           'x_bias': second_stage_weights['bias']}
            self.initial_model_weights = stage_three_initial_weights
        kwargs['print_complete_correlation'] = True
        kwargs['fit_parameters'] = self.fit_parameters['third_stage']
        kwargs['model_parameters'] = self.current_model_parameters
        self.current_loss_function_type = 'ipuf'
        self.model_class = MultiInterposePufNet
        third_fit_log = self.fit_single_reliability(training_input, training_output,
                                                    validation_input,
                                                    validation_output, **kwargs)

        third_stage_results = {'third_fit_log': third_fit_log}
        return third_stage_results

    def store_stage(self, training_input, training_output, validation_input, validation_output, stage_results,
                    stage_tag, store_in_out, store_kwargs, **kwargs):
        to_store_dict = {'stage_results': stage_results}
        if store_in_out:
            to_store_dict.update({'training_input': training_input, 'training_output': training_output,
                                  'validation_input': validation_input, 'validation_output': validation_output})
        if store_kwargs:
            to_store_dict.update({'kwargs': kwargs})
        joblib.dump(to_store_dict, stage_tag)

    def load_stage(self, stage_tag):
        return joblib.load(stage_tag)

    def compute_x_hypo_responses(self, model, challenges, actual_responses, puf_parameters, puf_weights):

        rand_puf = InterposePuf(**puf_parameters)
        ref_puf = InterposePuf(**puf_parameters)
        ref_puf.import_weights(puf_weights)
        _, x_responses, _ = ref_puf.compute_response(challenges, enable_noise=False,
                                                     input_is_feature_vector=True,
                                                     debug_output=True)

        x_resp_list = []
        x_hypo_acc = []
        for current_multi_puf in range(model.num_multi_pufs):
            model_weights = self.model.export_weights(export_single_multi_puf=current_multi_puf)
            rand_puf.import_weights(model_weights, import_x=False, import_y=True)
            # rand_puf.import_weights(puf_weights, import_x=True, import_y=True)
            x_one_response = np.ones_like(x_responses)

            # tmp_responses = rand_puf.compute_response(challenges, enable_noise=False, input_is_feature_vector=True,)
            rand_responses = rand_puf.compute_response(challenges, enable_noise=False, input_is_feature_vector=True,
                                                       external_x_response=x_one_response)
            # x_hypo_responses = (rand_responses == actual_responses).astype(int)
            if puf_parameters['num_x_xor_pufs'] % 2:
                x_hypo_responses = (rand_responses != actual_responses).astype(int)
            else:
                x_hypo_responses = (rand_responses == actual_responses).astype(int)
            x_resp_list.append(x_hypo_responses)
            x_hypo_acc.append(np.mean(x_hypo_responses == x_responses))

        print('Compute hypothetical x responses accuracy: {:s}'.format(str(np.array(x_hypo_acc))))

        return np.stack(x_resp_list).T, np.stack(x_hypo_acc).T

    def create_correlation_string(self, puf_weights, model_weights, **kwargs):
        model_parameters = kwargs['model_parameters']
        print_complete_correlation = kwargs.get('print_complete_correlation', False)
        if self.current_loss_function_type == 'ipuf':
            return super().create_correlation_string(puf_weights, model_weights, **kwargs)
        elif self.current_loss_function_type == 'xor':
            x_weights_shape = list(model_weights['weights'].shape)
            x_bias_shape = model_weights['bias'].shape
            x_weights_shape[2] += 1
            dummy_weights = {'y_weights': np.random.rand(*x_weights_shape),
                             'y_bias': np.random.rand(*x_bias_shape),
                             'x_weights': model_weights['weights'],
                             'x_bias': model_weights['bias']}
            result_string, result_corr = super().create_correlation_string_ipuf(puf_weights, dummy_weights,
                                                                                self.reference_ipuf_model_parameters,
                                                                                plot_y_correlation=False,
                                                                                print_both_halves=print_complete_correlation)
            # result_string = result_string + '\n' + self.create_correlation_string_model_to_orthogonal(model_weights, puf_weights, kwargs[ 'fit_parameters'][ 'orthogonal_weights'])
            return result_string, result_corr
        else:
            return '', None

    def create_correlation_string_model_to_orthogonal(self, model_weights, puf_weights, orthogonal_weights):
        multi_puf_index = 0
        result = 'MultiPUF: {:d}: '.format(multi_puf_index)
        for ortho_index in range(orthogonal_weights.shape[0]):
            corr = \
                np.corrcoef(model_weights['weights'][multi_puf_index, 0],
                            orthogonal_weights[multi_puf_index, ortho_index])[0, 1]
            result += 'x({:d}) -> o({:d}) ({:.2f}) | '.format(0, ortho_index, corr)

        pivot = 32
        num_stages = 64
        lookup = list(range(0, pivot)) + list(range(pivot + 1, num_stages + 1))
        lookup = list(range(0, pivot))
        y_weights = puf_weights['y_weights']
        result += '\n'
        for ortho_index in range(orthogonal_weights.shape[0]):
            for puf_index in range(y_weights.shape[0]):
                corr = np.corrcoef(y_weights[puf_index, lookup], orthogonal_weights[0, ortho_index, lookup])[0, 1]
                # corr = np.corrcoef(y_weights[0,lookup], orthogonal_weights[ortho_index])[0, 1]
                result += 'puf_y({:d}) -> o({:d}) ({:.2f}) | '.format(puf_index, ortho_index, corr)
            result += '\n'
        return result

    def eval_dl(self, model, dataloader, loss_batch, loss_func, fit_parameters):
        num_multi_pufs = model.num_multi_pufs
        loss_cum = 0
        total_num_samples = 0
        accuracies_cum = np.zeros((num_multi_pufs,))
        for batch_in, batch_out in dataloader:
            loss, num_items, model_output = loss_batch(model, loss_func, batch_in, batch_out)
            prediction = model_output[0]

            loss_cum += loss
            total_num_samples += num_items
            prediction = prediction > 0.5
            if fit_parameters.get('individual_puf_responses_during_optimization', False):
                puf_response = batch_out[:, :-1].type(torch.BoolTensor)
                current_accuracies = np.array(
                    [int(((prediction[multi_index]) == puf_response[:, multi_index]).sum()) for multi_index in
                     range(num_multi_pufs)])
            else:
                puf_response = batch_out[:, 0].type(torch.BoolTensor).squeeze()
                current_accuracies = np.array(
                    [int(((prediction[multi_index]) == puf_response).sum()) for multi_index in range(num_multi_pufs)])
            accuracies_cum += current_accuracies

        return loss_cum / total_num_samples, accuracies_cum / total_num_samples

    def invert_binary_vector(self, vector):
        return np.logical_not(vector).astype(int)


class ReliabilityComparisonMultiStageAttackIPUF(CombinedMultiStageAttackIPUF):

    def __init__(self, model_class, model_parameters, optim_parameters, fit_parameters):
        super().__init__(model_class, model_parameters, optim_parameters, fit_parameters)

    def create_loss_function(self, fit_parameter):
        return CombinedLossFunctions.create_xor_loss_function(fit_parameter)

    def create_model(self, **kwargs):
        model = self.model_class(**self.current_model_parameters)
        if self.initial_model_weights is not None:
            # TODO: This should be more generic
            model.import_weights(self.initial_model_weights, import_from_single_ipuf=False)
        return model

    def load_reference_data(self, **kwargs):
        fixed_puf_index = kwargs.get('fixed_puf_index', -1)
        if fixed_puf_index >= 0:
            puf_index_to_load = kwargs['puf_index']
        else:
            puf_index_to_load = fixed_puf_index
        puf_instance = kwargs['puf_instance']
        num_x_xors = puf_instance.num_x_xor_pufs
        num_y_xors = puf_instance.num_y_xor_pufs
        from puf_simulation_framework.attack_framework import AttackManager
        from puf_simulation_framework.attack_framework import PathManager
        path_manager = PathManager()
        path_manager.update_path_content(dict(base_path_name='./reference_data',
                                              topic_name='reference_results'))
        path_manager.update_path_content(dict(puf_type_name='interpose_puf',
                                              puf_identifier_name='({:d}, {:d})_stages_64'.format(num_x_xors,
                                                                                                  num_y_xors)))
        population_tag = kwargs['population_tag_to_load']

        attack_manager = AttackManager(path_manager, enable_logger=False)
        pop_info = attack_manager.load_population_info_by_tag(population_tag)
        loaded_attack_dicts = attack_manager.load_multiple_single_attacks_for_experiment(population_tag)
        loaded_dict = loaded_attack_dicts[puf_index_to_load]
        model_inputs = loaded_dict['model_inputs']
        model_outputs = loaded_dict['model_outputs']
        puf_weights = loaded_dict['puf_weights']

        return puf_weights, model_inputs, model_outputs

    def fit(self, training_input, training_output, validation_input, validation_output, **kwargs):

        use_loaded_responses = kwargs.get('use_loaded_responses', False)
        puf_weights, model_inputs, model_outputs = self.load_reference_data(**kwargs)
        current_puf = kwargs['puf_instance']
        current_puf.import_weights(puf_weights)
        training_input = model_inputs['training_noisy']
        validation_input = model_inputs['validation_noisy']
        if use_loaded_responses:
            training_output = model_outputs['training_noisy']
            validation_output = model_outputs['validation_noisy']

        else:
            responses = current_puf.compute_response(training_input, enable_noise=True,
                                                          input_is_feature_vector=True)
            reliability = current_puf.compute_reliability_response(training_input, 10,
                                                                        input_is_feature_vector=True,
                                                                        reliability_type='minus_abs')
            training_output = np.stack([responses, reliability], -1)
            responses = current_puf.compute_response(validation_input, enable_noise=True,
                                                          input_is_feature_vector=True)
            reliability = current_puf.compute_reliability_response(validation_input, 10,
                                                                        input_is_feature_vector=True,
                                                                        reliability_type='minus_abs')
            validation_output = np.stack([responses, reliability], -1)
            print('USING NEW PUF')

        self.current_loss_function_type = 'xor'
        self.model_class = MultiXorArbiterNet

        self.reference_ipuf_model_parameters = self.model_parameters['ipuf_model']

        # pivot = kwargs['puf_instance'].y_pivot
        # num_stages = kwargs['puf_instance'].num_stages
        # lookup = list(range(0, pivot)) + list(range(pivot + 1, num_stages + 1))
        self.current_model_parameters = self.model_parameters['xor_model']
        kwargs['model_parameters'] = self.current_model_parameters
        # inverted_orthogonal_y_weights[:, :, :pivot] *= -1
        # orthogonal_y_weights = np.concatenate([orthogonal_y_weights, inverted_orthogonal_y_weights], axis=1)
        # self.fit_parameters['second_stage']['orthogonal_weights'] = torch.tensor(orthogonal_y_weights)
        kwargs['fit_parameters'] = self.fit_parameters

        self.current_loss_function_type = 'xor'
        self.model_class = MultiXorArbiterNet
        # original_reference_puf_instance = kwargs['puf_instance']
        # The x reponses need to be inverted here? PUF model vs gradient model
        # second_stage_training_output = np.concatenate(
        #    [self.invert_binary_vector(training_x_responses), training_output[:, 1:2]], axis=-1)
        ##second_stage_validation_output = np.concatenate(
        #    [self.invert_binary_vector(validation_x_responses), validation_output[:, 1:2]], axis=-1)
        fit_log = self.fit_single_reliability(training_input, training_output,
                                              validation_input,
                                              validation_output, **kwargs)
        model_weights = self.model.export_weights()
        results = {'model_weights': model_weights,
                   'fit_log': fit_log}
        return results

    def create_correlation_string(self, puf_weights, model_weights, **kwargs):
        num_x_pufs = puf_weights['x_weights'].shape[0]
        num_y_pufs = puf_weights['x_weights'].shape[0]
        model_parameters = kwargs['model_parameters']
        print_complete_correlation = kwargs.get('print_complete_correlation', False)
        x_weights_shape = list(model_weights['weights'].shape)
        x_bias_shape = model_weights['bias'].shape
        x_weights_shape[2] += 1
        # dummy_weights = {'y_weights': model_weights['weights'][:, :num_y_pufs, :],
        #                 'y_bias': model_weights['bias'][:, :num_y_pufs ],
        #                 'x_weights': model_weights['weights'][:, num_y_pufs:, :],
        #                 'x_bias': model_weights['bias'][:, num_y_pufs:]}
        dummy_weights = {'y_weights': np.random.rand(*x_weights_shape),
                         'y_bias': np.random.rand(*x_bias_shape),
                         'x_weights': model_weights['weights'][:, num_y_pufs:, :],
                         'x_bias': model_weights['bias'][:, num_y_pufs:]}
        result_string, result_corr = self.create_correlation_string_ipuf(puf_weights, model_weights,
                                                                         self.reference_ipuf_model_parameters,
                                                                         plot_y_correlation=False,
                                                                         print_both_halves=print_complete_correlation)
        # result_string = result_string + '\n' + self.create_correlation_string_model_to_orthogonal(model_weights, puf_weights, kwargs[ 'fit_parameters'][ 'orthogonal_weights'])
        return result_string, result_corr

    def create_correlation_string_ipuf(self, puf_weights, model_weights, model_parameters, plot_y_correlation=True,
                                       print_both_halves=False):
        num_stages = model_parameters['num_stages']
        pivot = model_parameters['y_pivot']
        num_multi_pufs = model_parameters['num_multi_pufs']
        # änum_x_pufs = model_parameters['num_x_xors']
        # num_y_pufs = model_parameters['num_y_xors']
        num_model_weights = model_weights['weights'].shape[1]
        first_half_model_to_puf_correlation = CorrelationUtil.compute_ipuf_multi_puf_to_xor_model_correlations(
            puf_weights,
            model_weights,
            num_stages,
            pivot, 'first')
        second_half_model_to_puf_correlation = CorrelationUtil.compute_ipuf_multi_puf_to_xor_model_correlations(
            puf_weights,
            model_weights,
            num_stages,
            pivot, 'second')

        result_string = ''
        for multi_puf in range(num_multi_pufs):
            result_string += '\n#{:d}'.format(multi_puf)
            for x_puf in range(num_model_weights):
                result_string += '| x{:d} hit {:s} ({: .2f})'.format(x_puf,
                                                                     first_half_model_to_puf_correlation['model_x_hit'][
                                                                         multi_puf, x_puf],
                                                                     first_half_model_to_puf_correlation[
                                                                         'model_x_best_correlation'][multi_puf,
                                                                                                     x_puf])

        correlation_result = {'first_half': first_half_model_to_puf_correlation,
                              'second_half': second_half_model_to_puf_correlation}

        return result_string, correlation_result
