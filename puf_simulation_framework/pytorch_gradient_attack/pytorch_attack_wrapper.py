import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import time
from puf_simulation_framework.attack_framework import AttackObject
from .attack_util import PytorchAttackUtil
from puf_simulation_framework.pufs import PufUtil


class PytorchWrapper(AttackObject):

    def __init__(self, model_class, model_parameters, optim_parameters, fit_parameters):
        self.model_class = model_class
        self.model_parameters = model_parameters
        self.optim_parameters = optim_parameters
        self.fit_parameters = fit_parameters
        self.model = None
        self.model_tensor_weights = None

    def get_summary_dict(self):
        summary = {'model_class_full': str(self.model_class),
                   'model_class': self.model_class.__name__,
                   'model_parameters': self.model_parameters,
                   'optim_parameters': self.optim_parameters,
                   'fit_parameters': self.fit_parameters}
        return summary

    def arrays_to_dataset(self, challenges, responses, dtype=torch.float):
        # return TensorDataset(torch.Tensor(challenges, dtype=torch.int8), torch.Tensor(responses, dtype=float))
        # return TensorDataset(torch.as_tensor(challenges, dtype=torch.int8), torch.as_tensor(responses, dtype=float))
        return TensorDataset(torch.as_tensor(challenges, dtype=dtype), torch.as_tensor(responses, dtype=dtype))

    def loss_batch(self, model, loss_func, xb, yb, opt=None):
        model_output = model(xb).squeeze()
        #model_weights = model.export_tensor_weights()
        model_weights = None
        loss = loss_func(model_output, yb, model_weights)
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(xb), model_output

    def eval_dl(self, model, dataloader, loss_batch, loss_func):
        loss_cum = 0
        total_num_samples = 0
        accuracies_cum = 0
        for batch_in, batch_out in dataloader:
            loss, num_items, prediction = loss_batch(model, loss_func, batch_in, batch_out)

            loss_cum += loss
            total_num_samples += num_items
            prediction = prediction.squeeze() > 0.5
            batch_out = batch_out.type(torch.BoolTensor).squeeze()
            accuracies_cum += int(((prediction) == batch_out).sum())

        return loss_cum / total_num_samples, accuracies_cum / total_num_samples

    def create_optimizer(self, model):
        optimizer_name = self.optim_parameters['optimizer_name']
        optimizer_parameters = self.optim_parameters['optimizer_parameters']

        if optimizer_name == 'Adadelta':
            opt = torch.optim.Adadelta(model.parameters())
        elif optimizer_name == 'SGD':
            opt = torch.optim.SGD(model.parameters(), **optimizer_parameters)
        elif optimizer_name == 'Adam':
            opt = torch.optim.Adam(model.parameters(), **optimizer_parameters)
        elif optimizer_name == 'LBFGS':
            opt = torch.optim.LBFGS(model.parameters(), **optimizer_parameters)
        elif optimizer_name == 'RMSprop':
            opt = torch.optim.RMSprop(model.parameters(), **optimizer_parameters)
        elif optimizer_name == 'Rprop':
            opt = torch.optim.Rprop(model.parameters(), **optimizer_parameters)
        else:
            raise ValueError('Unknown optimizer: {:s}'.format(optimizer_name))

        return opt

    def parse_string_to_dtype(self, dtype_string):
        if dtype_string == 'float32' or dtype_string == '':
            return torch.float32
        elif dtype_string == 'float64':
            return torch.float64
        else:
            raise ValueError

    def create_loss_function(self, fit_parameter):
        return lambda x, y, model_weights: F.binary_cross_entropy(x,y)


    def create_data_loaders(self, training_input, training_output, validation_input, validation_output, fit_parameters):
        dtype_string = fit_parameters.get('optimizer_dtype', '')
        dtype = self.parse_string_to_dtype(dtype_string)
        training_dataset = self.arrays_to_dataset(training_input, training_output, dtype=dtype)
        validation_dataset = self.arrays_to_dataset(validation_input, validation_output, dtype=dtype)
        batch_size = fit_parameters['batch_size']
        train_dl = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        validation_dl = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        return train_dl, validation_dl

    def fit(self, training_input, training_output, validation_input, validation_output, **kwargs):

        if self.model is not None:
            del self.model
            self.model = None

        if kwargs.get('convert_input_to_feature_vector', False):
            training_input = PufUtil.challenge_to_feature_vector(training_input)
            validation_input = PufUtil.challenge_to_feature_vector(validation_input)

        self.model = self.model_class(**self.model_parameters)

        loss_func = self.create_loss_function(self.fit_parameters)
        opt = self.create_optimizer(self.model)
        train_dl, validation_dl = self.create_data_loaders(training_input, training_output, validation_input,
                                                           validation_output, self.fit_parameters)

        num_epochs = self.fit_parameters['num_epochs']
        verbose = self.fit_parameters.get('verbose', False)

        per_epoch_log = []
        for epoch in range(num_epochs):

            self.model.train()
            for train_in, train_out in train_dl:
                self.loss_batch(self.model, loss_func, train_in, train_out, opt=opt)

            self.model.eval()
            val_loss, val_acc = self.eval_dl(self.model, validation_dl, self.loss_batch, loss_func)
            train_loss, train_acc = self.eval_dl(self.model, train_dl, self.loss_batch, loss_func)
            new_log_entry = self.evaluate_epoch(train_loss, train_acc, val_loss, val_acc, epoch, num_epochs, verbose)
            per_epoch_log.append(new_log_entry)

        final_model_weights = self.model.export_weights()
        fit_log = {'per_epoch_log': per_epoch_log,
                   'final_model_weights': final_model_weights}
        return fit_log

    def evaluate_epoch(self, train_loss, train_acc, val_loss, val_acc, current_epoch, total_num_epochs, verbose):
        result = {'validation_loss': val_loss,
                  'validation_accuracy': val_acc,
                  'training_loss': train_loss,
                  'training_accuracy': train_acc}
        if verbose:
            print(
                "Epoch #{:d}/{:d}: Train loss: {:.2f}, Train acc: {:.2f}, Val loss: {:.2f}, Val acc: {:.2f}".format(
                    current_epoch, total_num_epochs, train_loss, train_acc,
                    val_loss, val_acc))
        return result

    def predict(self, input, round_result=True):
        if self.model is None:
            raise Exception
        self.model.eval()
        result = self.model(torch.Tensor(input)).data.numpy()
        if round_result:
            result = np.round(result)
        return result


class MultiPytorchWrapper(PytorchWrapper):

    def eval_dl(self, model, dataloader, loss_batch, loss_func):
        num_multi_pufs = model.num_multi_pufs
        loss_cum = 0
        total_num_samples = 0
        accuracies_cum = np.zeros((num_multi_pufs,))
        for batch_in, batch_out in dataloader:
            loss, num_items, prediction = loss_batch(model, loss_func, batch_in, batch_out)

            loss_cum += loss
            total_num_samples += num_items
            prediction = prediction > 0.5
            batch_out = batch_out.type(torch.BoolTensor).squeeze()
            current_accuracies = np.array([int(((prediction[multi_index]) == batch_out).sum()) for multi_index in range(num_multi_pufs)])
            accuracies_cum += current_accuracies

        return loss_cum / total_num_samples, accuracies_cum / total_num_samples

    def evaluate_epoch(self, train_loss, train_acc, val_loss, val_acc, current_epoch, total_num_epochs, verbose):
        self.best_model = np.argmax(val_acc)
        result = {'validation_loss': val_loss,
                  'validation_accuracy': val_acc,
                  'training_loss': train_loss,
                  'training_accuracy': train_acc,
                  'best_model_index': self.best_model}
        # 
        if verbose:
            accuracy_string = ''
            index = 0
            for t_acc, v_acc in zip(train_acc.tolist(), val_acc.tolist()):
                accuracy_string += 'Multi #{:d}: Train acc {:.2f}, Val acc {:.2f} | '.format(index, t_acc, v_acc)
                index += 1
            print('Epoch #{:d}/{:d}: {:s}'.format(current_epoch, total_num_epochs, accuracy_string))
        return result

    def predict(self, input, round_result=True):
        if self.model is None:
            raise Exception
        self.model.eval()
        result = self.model(torch.Tensor(input)).data.numpy()[self.best_model] # ToDo: Make sure that best_model always exists
        if round_result:
            result = np.round(result)
        return result

    def loss_batch(self, model, loss_func, xb, yb, opt=None):
        #model_output = model(xb).squeeze()
        model_output = model(xb)
        #model_weights = model.export_tensor_weights()
        model_weights = None
        loss = loss_func(model_output, yb, model_weights)
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(xb), model_output

    def create_loss_function(self, fit_parameter):
        def multi_binary_cross_entropy(x, y, model_weights):
            loss_list = []
            for index in range(x.shape[0]):
                loss_list.append(F.binary_cross_entropy(x[index], y))
            return torch.sum(torch.stack(loss_list))
        return multi_binary_cross_entropy

    def create_loss_function_ipuf(self, fit_parameter, orthogonal_weights=None):
        pivot = self.model_parameters['y_pivot']  # TODO: refactor

        def multi_puf_pearson_loss(x, y, model_weights, orthogonal_weights=orthogonal_weights, pivot=pivot):
            x_y_cost_scale = 0.05
            # x_y_cost_scale = 0.0
            # x_y_cost_scale = 5000
            multi_loss_list = []
            for current_multi_puf in range(x.shape[0]):
                cost_sum = F.binary_cross_entropy(x[current_multi_puf], y)

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

            return sum_loss

        return multi_puf_pearson_loss


class PytorchReliabilityWrapper(PytorchWrapper):

    def compute_correlation(self, data_input, data_output):
        prediction = self.predict(data_input)
        return np.corrcoef(prediction, data_output)[0, 1]

    def create_loss_function(self, fit_parameter, orthogonal_weights=None):

        def combined_loss(x, y, model_weights, orthogonal_weights=orthogonal_weights):
            cost_sum = PytorchAttackUtil.pearson_loss(x, y)

            if orthogonal_weights is not None:
                m_w = model_weights['weights'].squeeze()
                o_w = orthogonal_weights['weights']
                ortho_scale = x.shape[0] / 400

                additional_cost = []
                for index in range(o_w.shape[0]):
                    length = torch.norm(m_w) * torch.norm(o_w[index])
                    additional_cost.append(torch.abs(torch.dot(m_w, o_w[index])) / length)

                additional_cost_sum = torch.sum(torch.stack(additional_cost).flatten()) * ortho_scale
                cost_sum += additional_cost_sum

            return cost_sum

        return combined_loss

    def loss_batch(self, model, loss_func, xb, yb, opt=None):
        xb_float = xb.float()
        model_output = model(xb_float).squeeze()
        model_weights = model.export_tensor_weights()
        # model_weights = self.model_tensor_weights
        loss = loss_func(model_output, yb, model_weights)
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(xb), model_output

    def fit_single_reliability(self, training_input, training_output, validation_input, validation_output, **kwargs):
        current_puf = kwargs['puf_instance']
        puf_parameter = current_puf.export_weights()
        puf_weights = puf_parameter['weights']
        orthogonal_weights = kwargs.get('orthogonal_weights', None)

        if self.model is not None:
            del self.model
            self.model = None

        self.model = self.model_class(**self.model_parameters)

        # Get a reference to the model weights
        self.model_tensor_weights = self.model.export_weights()

        loss_func = self.create_loss_function(self.fit_parameters, orthogonal_weights=orthogonal_weights)
        opt = self.create_optimizer(self.model)
        train_dl, validation_dl = self.create_data_loaders(training_input, training_output,
                                                           validation_input,
                                                           validation_output)
        num_epochs = self.fit_parameters['num_epochs']
        verbose = self.fit_parameters.get('verbose', False)

        per_epoch_log = []
        for epoch in range(num_epochs):

            self.model.train()
            for train_in, train_out in train_dl:
                self.loss_batch(self.model, loss_func, train_in, train_out, opt=opt)

            self.model.eval()
            val_loss, val_acc = self.eval_dl(self.model, validation_dl, self.loss_batch, loss_func)
            train_loss, train_acc = self.eval_dl(self.model, train_dl, self.loss_batch, loss_func)
            per_epoch_log.append({'validation_loss': val_loss,
                                  'validation_accuracy': val_acc,
                                  'training_loss': train_loss,
                                  'training_accuracy': train_acc})
            if verbose:
                model_param = self.model.export_weights()
                model_weights = model_param['weights'].squeeze()

                correlation_string = ''
                for current_xor in range(puf_weights.shape[0]):
                    corr = np.corrcoef(puf_weights[current_xor], model_weights)
                    correlation_string += 'Corr PUF #{:d}: {:f}, '.format(current_xor, corr[0, 1])

                print("Epoch #{:d}/{:d}: Train loss: {:.5f}, {:s}".format(
                    epoch, num_epochs, train_loss, correlation_string))
        fit_log = {'per_epoch_log': per_epoch_log}

    def fit(self, training_input, training_output, validation_input, validation_output, **kwargs):
        training_reliability_output = training_output[:, 1]
        validation_reliability_output = validation_output[:, 1]

        target_num_xor = self.model_parameters['num_xors']

        # for the training, the model needs to return
        self.model_parameters['output_type'] = 'abs_raw'
        self.model_parameters['num_xors'] = 1
        self.fit_parameters['num_epochs'] = 100

        self.fit_single_reliability(training_input, training_reliability_output, validation_input,
                                    validation_reliability_output, **kwargs)
        ortho_weight = self.model.export_tensor_weights()
        kwargs['orthogonal_weights'] = ortho_weight

        for current_xor in range(1, target_num_xor):
            print('----------------------------------------------------')
            self.fit_parameters['num_epochs'] = 200
            self.fit_single_reliability(training_input, training_reliability_output, validation_input,
                                        validation_reliability_output, **kwargs)
            ortho_weight = self.model.export_tensor_weights()
            tmp_ortho_weights = kwargs['orthogonal_weights']['weights']
            tmp_ortho_bias = kwargs['orthogonal_weights']['bias']
            tmp_ortho_weights = torch.cat([tmp_ortho_weights, ortho_weight['weights']], axis=0)
            tmp_ortho_bias = torch.cat([tmp_ortho_bias, ortho_weight['bias']], axis=0)
            kwargs['orthogonal_weights']['weights'] = tmp_ortho_weights
            kwargs['orthogonal_weights']['bias'] = tmp_ortho_bias

        # for cnt in range(6):
        #     self.fit_single_reliability(training_input, training_reliability_output, validation_input, validation_reliability_output, **kwargs)

        # return fit_log

    def predict_raw(self, input):
        if self.model is None:
            raise Exception
        self.model.eval()
        return self.model(torch.Tensor(input)).data.numpy()


class PytorchIPUFReliability(PytorchReliabilityWrapper):

    def create_loss_function(self, fit_parameter, orthogonal_weights=None):

        def pearson_loss(x, y, model_weights, orthogonal_weights=orthogonal_weights):
            cost_sum = PytorchAttackUtil.pearson_loss(x, y)

            x_weights = model_weights['x_weights'].squeeze()
            y_weights = model_weights['y_weights'].squeeze()
            pivot = 32
            num_stages = 64
            lookup = list(range(0, pivot)) + list(range(pivot + 1, num_stages + 1))

            # force x and y puf to be different
            # x_y_cost_scale = x.shape[0]/400
            x_y_cost_scale = 0.1
            cost_sum += torch.abs(torch.dot(x_weights, y_weights[lookup])) * x_y_cost_scale

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

            return cost_sum

        return pearson_loss

    def fit_single_reliability(self, training_input, training_output, validation_input, validation_output, **kwargs):
        current_puf = kwargs['puf_instance']
        puf_parameter = current_puf.export_weights()
        puf_x_weights = puf_parameter['x_weights']
        puf_y_weights = puf_parameter['y_weights']
        pivot = self.model_parameters['y_pivot']
        num_stages = self.model_parameters['num_stages']
        orthogonal_weights = kwargs.get('orthogonal_weights', None)

        if self.model is not None:
            del self.model
            self.model = None

        self.model = self.model_class(**self.model_parameters)
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
            train_correlation = self.compute_correlation(training_input, training_output)
            validation_correlation = self.compute_correlation(validation_input, validation_output)

            # if verbose:
            model_param = self.model.export_weights()
            model_x_weights = model_param['x_weights'].squeeze()
            model_y_weights = model_param['y_weights'].squeeze()

            correlation_string = ''
            corr_puf_x_to_model_x = []
            corr_puf_y_to_model_y = []
            for current_xor in range(puf_x_weights.shape[0]):
                corr = np.corrcoef(puf_x_weights[current_xor], model_x_weights)[0, 1]
                correlation_string += 'X Corr PUF #{:d}: {:f}, '.format(current_xor, corr)
                corr_puf_x_to_model_x.append(corr)
            for current_xor in range(puf_y_weights.shape[0]):
                corr = np.corrcoef(puf_y_weights[current_xor], model_y_weights)[0, 1]
                correlation_string += 'Y Corr PUF #{:d}: {:f}, '.format(current_xor, corr)
                corr_puf_y_to_model_y.append(corr)

            correlation_string += '\n'

            lookup = list(range(0, pivot)) + list(range(pivot + 1, num_stages + 1))

            corr_puf_y_to_model_x = []
            corr_puf_x_to_model_y = []
            for current_xor in range(puf_y_weights.shape[0]):
                corr = np.corrcoef(puf_y_weights[current_xor, lookup], model_x_weights)[0, 1]
                correlation_string += 'X->Y Corr PUF #{:d}: {:f}, '.format(current_xor, corr)
                corr_puf_y_to_model_x.append(corr)
            for current_xor in range(puf_x_weights.shape[0]):
                corr = np.corrcoef(puf_x_weights[current_xor], model_y_weights[lookup])[0, 1]
                correlation_string += 'Y->X Corr PUF #{:d}: {:f}, '.format(current_xor, corr)
                corr_puf_x_to_model_y.append(corr)

            corr = np.corrcoef(model_x_weights, model_y_weights[lookup])[0, 1]
            correlation_string += 'net_Y<->net_X {:f}, '.format(corr)

            print("Epoch #{:d}/{:d}: Train loss: {:.5f}, Train Corr: {:.5f}, Vali Corr: {:.5f}, {:s}".format(
                epoch, num_epochs, train_loss, train_correlation, validation_correlation, correlation_string))

            per_epoch_log.append({'validation_loss': val_loss,
                                  'validation_accuracy': val_acc,
                                  'training_loss': train_loss,
                                  'training_accuracy': train_acc,
                                  'corr_puf_x_to_model_x': corr_puf_x_to_model_x,
                                  'corr_puf_y_to_model_y': corr_puf_y_to_model_y,
                                  'corr_puf_x_to_model_y': corr_puf_x_to_model_y,
                                  'corr_puf_y_to_model_x': corr_puf_y_to_model_x,
                                  })
        duration = time.time() - start_time

        fit_log = {'per_epoch_log': per_epoch_log,
                   'final_training_correlation': train_correlation,
                   'final_validation_correlation': validation_correlation,
                   'model_weights': model_param,
                   'fit_duration': duration}
        return fit_log

    def fit(self, training_input, training_output, validation_input, validation_output, **kwargs):
        training_reliability_output = training_output[:, 1]
        validation_reliability_output = validation_output[:, 1]
        training_feature_vectors = PufUtil.challenge_to_feature_vector(training_input)
        validation_feature_vectors = PufUtil.challenge_to_feature_vector(validation_input)

        # for the first stage, fix a (1,1)-IPUF as the model to fit
        # self.model_parameters['output_type'] = 'y_abs_raw_x_abs_raw'
        self.model_parameters['output_type'] = 'abs_raw'
        self.model_parameters['num_x_xors'] = 1
        self.model_parameters['num_y_xors'] = 1
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


#
class PytorchIPUFReliabilityWithXorModel(PytorchReliabilityWrapper):

    def create_loss_function(self, fit_parameter,  orthogonal_weights=None):

        def pearson_loss(x, y, model_weights, orthogonal_weights=orthogonal_weights):
            cost_sum = PytorchAttackUtil.pearson_loss(x, y)

            if orthogonal_weights is not None:
                m_w = model_weights['weights'].squeeze()
                o_w = orthogonal_weights['weights']
                ortho_scale = x.shape[0] / 400

                additional_cost = []
                num_constraints = o_w.shape[0]
                for index in range(num_constraints):
                    length = torch.norm(m_w) * torch.norm(o_w[index])
                    additional_cost.append(torch.abs(torch.dot(m_w, o_w[index])) / length)

                additional_cost_sum = torch.sum(torch.stack(additional_cost).flatten()) * (
                            ortho_scale / num_constraints)
                cost_sum += additional_cost_sum

            return cost_sum

        return pearson_loss

    def loss_batch(self, model, loss_func, xb, yb, opt=None):
        model_output = model(xb).squeeze()
        model_weights = model.export_tensor_weights()
        loss = loss_func(model_output, yb, model_weights)
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(xb), model_output

    def fit_single_reliability(self, training_input, training_output, validation_input, validation_output, **kwargs):
        current_puf = kwargs['puf_instance']
        puf_parameter = current_puf.export_weights()
        puf_x_weights = puf_parameter['x_weights']
        puf_y_weights = puf_parameter['y_weights']
        pivot = kwargs['y_pivot']
        num_stages = self.model_parameters['num_stages']
        orthogonal_weights = kwargs.get('orthogonal_weights', None)

        if self.model is not None:
            del self.model
            self.model = None

        self.model = self.model_class(**self.model_parameters)

        loss_func = self.create_loss_function(self.fit_parameters, orthogonal_weights=orthogonal_weights)
        opt = self.create_optimizer(self.model)
        train_dl, validation_dl = self.create_data_loaders(training_input, training_output,
                                                           validation_input,
                                                           validation_output)
        num_epochs = self.fit_parameters['num_epochs']
        verbose = self.fit_parameters.get('verbose', False)

        per_epoch_log = []
        for epoch in range(num_epochs):

            self.model.train()
            for train_in, train_out in train_dl:
                self.loss_batch(self.model, loss_func, train_in, train_out, opt=opt)

            self.model.eval()
            val_loss, val_acc = self.eval_dl(self.model, validation_dl, self.loss_batch, loss_func)
            train_loss, train_acc = self.eval_dl(self.model, train_dl, self.loss_batch, loss_func)
            train_correlation = self.compute_correlation(training_input, training_output)
            validation_correlation = self.compute_correlation(validation_input, validation_output)
            per_epoch_log.append({'validation_loss': val_loss,
                                  'validation_accuracy': val_acc,
                                  'training_loss': train_loss,
                                  'training_accuracy': train_acc,
                                  'training_correlation': train_correlation,
                                  'validation_correlation': validation_correlation})
            model_param = self.model.export_weights()
            if verbose:

                model_weights = model_param['weights'].squeeze()
                # model_y_weights = model_param['y_weights'].squeeze()

                correlation_string = ''
                for current_xor in range(puf_x_weights.shape[0]):
                    corr = np.corrcoef(puf_x_weights[current_xor], model_weights)
                    correlation_string += 'X Corr PUF #{:d}: {:f}, '.format(current_xor, corr[0, 1])
                # for current_xor in range(puf_y_weights.shape[0]):
                #     corr = np.corrcoef(puf_y_weights[current_xor], model_weights)
                #     correlation_string += 'Y Corr PUF #{:d}: {:f}, '.format(current_xor, corr[0, 1])

                correlation_string += '\n'

                lookup = list(range(0, pivot)) + list(range(pivot + 1, num_stages + 1))

                for current_xor in range(puf_y_weights.shape[0]):
                    corr = np.corrcoef(puf_y_weights[current_xor, lookup], model_weights)
                    correlation_string += 'X->Y Corr PUF #{:d}: {:f}, '.format(current_xor, corr[0, 1])
                # for current_xor in range(puf_x_weights.shape[0]):
                #     corr = np.corrcoef(puf_x_weights[current_xor], model_y_weights[lookup])
                #     correlation_string += 'Y->X Corr PUF #{:d}: {:f}, '.format(current_xor, corr[0, 1])

                correlation_string += '\n Model Self Corr: {:s}'.format(str(np.corrcoef(model_weights).flatten()))

                print("Epoch #{:d}/{:d}: Train loss: {:.5f}, Train corr: {:.5f}, Val corr: {:.5f}, {:s}".format(
                    epoch, num_epochs, train_loss, train_correlation, validation_correlation, correlation_string))
        fit_log = {'per_epoch_log': per_epoch_log,
                   'final_trainining_correlation': train_correlation,
                   'final_validation_correlation': validation_correlation,
                   'model_weights': model_param}
        return fit_log

    def fit(self, training_input, training_output, validation_input, validation_output, **kwargs):
        training_reliability_output = training_output[:, 1]
        validation_reliability_output = validation_output[:, 1]

        # for the training, the model needs to return
        self.model_parameters['output_type'] = 'abs_raw'
        self.model_parameters['num_xors'] = 2
        self.fit_parameters['num_epochs'] = 100
        num_collections = 10

        fit_logs = []
        tmp = self.fit_single_reliability(training_input, training_reliability_output, validation_input,
                                          validation_reliability_output, **kwargs)
        fit_logs.append(tmp)
        ortho_weight = None  # self.model.export_tensor_weights()
        kwargs['orthogonal_weights'] = ortho_weight
        target_num_xor = self.model_parameters['num_y_xors']
        for current_xor in range(num_collections):
            print('----------------------------------------------------')
            print('Collection #{:d}'.format(current_xor))
            self.fit_parameters['num_epochs'] = 200
            tmp = self.fit_single_reliability(training_input, training_reliability_output, validation_input,
                                              validation_reliability_output, **kwargs)
            fit_logs.append(tmp)
            # ortho_weight = self.model.export_tensor_weights()
            # tmp_ortho_weights = kwargs['orthogonal_weights']['weights']
            # tmp_ortho_bias = kwargs['orthogonal_weights']['bias']
            # tmp_ortho_weights = torch.cat([tmp_ortho_weights, ortho_weight['weights']], axis=0)
            # tmp_ortho_bias = torch.cat([tmp_ortho_bias, ortho_weight['bias']], axis=0)
            # kwargs['orthogonal_weights']['weights'] = tmp_ortho_weights
            # kwargs['orthogonal_weights']['bias'] = tmp_ortho_bias

        print()

        # for cnt in range(6):
        #     self.fit_single_reliability(training_input, training_reliability_output, validation_input, validation_reliability_output, **kwargs)

        # return fit_log

    def predict_raw(self, input):
        if self.model is None:
            raise Exception
        self.model.eval()
        return self.model(torch.Tensor(input)).data.numpy()
