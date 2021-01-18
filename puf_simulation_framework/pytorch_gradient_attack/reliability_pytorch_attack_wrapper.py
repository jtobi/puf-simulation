import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import time
from puf_simulation_framework.attack_framework import AttackObject
from .attack_util import PytorchAttackUtil
from puf_simulation_framework.pufs import PufUtil
from .pytorch_attack_wrapper import PytorchWrapper


class PytorchIPUFReliability(PytorchReliabilityWrapper):

    def create_loss_function(self, orthogonal_weights=None):

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

        loss_func = self.create_loss_function(orthogonal_weights=orthogonal_weights)
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

        fit_log = {'per_epoch_log': per_epoch_log,
                   'final_training_correlation': train_correlation,
                   'final_validation_correlation': validation_correlation,
                   'model_weights': model_param}
        return fit_log


    def predict_raw(self, input):
        if self.model is None:
            raise Exception
        self.model.eval()
        return self.model(torch.Tensor(input)).data.numpy()