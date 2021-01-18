import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np

from puf_simulation_framework.pufs import InterposePuf
from puf_simulation_framework.pufs import PufUtil


class XorArbiterNet(nn.Module):

    def __init__(self, num_xors, num_stages, output_type='sigmoid'):
        super().__init__()
        self.num_xors = num_xors
        self.num_stages = num_stages

        self.arbiter_layer = nn.ModuleList([nn.Linear(num_stages, 1, bias=True) for _ in range(num_xors)])
        self.arbiter_batch_norm = nn.BatchNorm1d(num_xors)

        self.output_type = output_type

    def forward(self, x):
        raw_arbiter_out = [lin_layer(x) for lin_layer in self.arbiter_layer]
        raw_arbiter_out = torch.cat(raw_arbiter_out, dim=-1)
        raw_arbiter_out = self.arbiter_batch_norm(raw_arbiter_out)
        raw_out = torch.prod(raw_arbiter_out, -1)

        if self.output_type == 'raw':
            output = raw_out
        elif self.output_type == 'abs_raw':
            output = torch.abs(raw_out)
        elif self.output_type == 'sigmoid':
            output = torch.sigmoid(raw_out)
        else:
            raise ValueError()
        return output

    def export_weights(self):
        weights = []
        bias = []
        for layer in self.arbiter_layer:
            weights.append(layer.weight.data.numpy().squeeze())
            bias.append(layer.bias.data.numpy())

        result = {'weights': np.stack(weights, 0),
                  'bias': np.stack(bias, 0)}
        return result

    def export_tensor_weights(self):
        weights = []
        bias = []
        for layer in self.arbiter_layer:
            weights.append(layer.weight.squeeze())
            bias.append(layer.bias)

        result = {'weights': torch.stack(weights, 0),
                  'bias': torch.stack(bias, 0)}
        return result

    # def import_weights(self, weight_dict):
    #
    #     x_weights = weight_dict['x_weights']
    #     x_bias = weight_dict['x_bias']
    #
    #     for puf_index in range(x_weights.shape[0]):
    #         self.x_arbiter_layer[puf_index].weight.data = torch.Tensor(x_weights[puf_index]).unsqueeze(0)
    #         self.x_arbiter_layer[puf_index].bias.data = torch.Tensor(x_bias[puf_index])
    #
    #     y_weights = weight_dict['y_weights']
    #     y_bias = weight_dict['y_bias']
    #
    #     for puf_index in range(y_weights.shape[0]):
    #         self.y_arbiter_layer[puf_index].weight.data = torch.Tensor(np.concatenate((y_weights[puf_index], y_bias[puf_index]))).unsqueeze(0)
    #         #self.y_arbiter_layer[puf_index].weight.data = torch.Tensor(y_weights[puf_index]).unsqueeze(0)
    #         #self.y_arbiter_layer[puf_index].bias.data = torch.Tensor(y_bias[puf_index])

class MultiXorArbiterNet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.num_xors = kwargs.get('num_xors')
        self.num_stages = kwargs.get('num_stages')
        self.num_multi_pufs = kwargs.get('num_multi_pufs')

        self.arbiter_layer = nn.Linear(self.num_stages, self.num_xors*self.num_multi_pufs, bias=True)
        self.arbiter_batch_norm = nn.BatchNorm1d(self.num_xors*self.num_multi_pufs)

        self.output_type = kwargs.get('output_type', 'sigmoid')
        self.include_per_arbiter_batch_norm = kwargs.get('include_per_arbiter_batch_norm', False)

    def forward(self, x):
        raw_arbiter_out = self.arbiter_layer(x)
        if self.include_per_arbiter_batch_norm:
            raw_arbiter_out = self.arbiter_batch_norm(raw_arbiter_out)
        raw_arbiter_out = raw_arbiter_out.reshape([-1, self.num_multi_pufs, self.num_xors])
        raw_out = torch.prod(raw_arbiter_out, -1)

        if self.output_type == 'raw':
            output = raw_out.transpose(0,1)
        elif self.output_type == 'abs_raw':
            output = torch.abs(raw_out).transpose(0,1)
        elif self.output_type == 'sigmoid':
            output = torch.sigmoid(raw_out).transpose(0,1)
        elif self.output_type == 'sigmoid_and_abs_raw':
            reliability_output = torch.abs(raw_arbiter_out).reshape([-1, self.num_multi_pufs, self.num_xors])
            reliability_output = reliability_output.transpose(0,1)
            reliability_output = reliability_output.transpose(1,2)
            output = (torch.sigmoid(raw_out).transpose(0,1), reliability_output)
        else:
            raise ValueError()
        return output

    def export_weights(self):
        weights = self.arbiter_layer.weight.data.numpy().reshape(self.num_multi_pufs, self.num_xors, -1)
        bias = self.arbiter_layer.bias.data.numpy().reshape(self.num_multi_pufs, self.num_xors)

        result = {'weights': weights,
                  'bias': bias}
        return result

    def export_tensor_weights(self):
        weights = self.arbiter_layer.weight.reshape(self.num_multi_pufs, self.num_xors, self.num_stages)
        bias = self.arbiter_layer.bias.reshape(self.num_multi_pufs, self.num_xors)

        result = {'weights': weights,
                  'bias': bias}
        return result

    def compute_single_puf_responses(self, feature_vectors):
        raw_arbiter_out = self.arbiter_layer(torch.tensor(feature_vectors, dtype=torch.float))
        raw_arbiter_out = raw_arbiter_out.reshape([-1, self.num_multi_pufs, self.num_xors])
        output = torch.sign(raw_arbiter_out)
        output = (output.data.numpy() > 0).astype('int')
        return output.transpose(1,2,0)
    


class InterposePufNet(nn.Module):

    def __init__(self, num_x_xors, num_y_xors, num_stages, y_pivot, output_type='sigmoid', input_is_feature_vector=False):
        super().__init__()
        self.num_x_xors = num_x_xors
        self.num_y_xors = num_y_xors
        self.num_stages = num_stages
        self.y_pivot = y_pivot

        self.x_arbiter_layer = nn.ModuleList([nn.Linear(num_stages, 1, bias=True) for _ in range(num_x_xors)])
        self.y_arbiter_layer = nn.ModuleList([nn.Linear(num_stages + 2, 1, bias=False) for _ in range(num_y_xors)])
        self.x_batch_norm = nn.BatchNorm1d(1)

        self.debug_counter = 0
        self.per_batch_mean = []
        self.per_batch_var = []

        self.output_type = output_type
        self.input_is_feature_vector = input_is_feature_vector

        self.y_feature_vector_matrix = None

    def forward(self, x):

        if not self.input_is_feature_vector:
            # Convert input challenge bits ([0,1]) to feature vector bits [-1, 1]
            feature_vectors = []
            for current_index in range(x.shape[1]):
                feature_vectors.append(torch.prod(1 - 2 * x[:, current_index:], axis=-1))
            feature_vector_matrix = torch.stack(feature_vectors, 1)
        else:
            feature_vector_matrix = x

        # Compute x-PUF output
        raw_x_arbiter_out = [lin_layer(feature_vector_matrix) for lin_layer in self.x_arbiter_layer]
        raw_x_xor_arbiter_out = torch.cat(raw_x_arbiter_out, dim=-1)
        raw_x_out = torch.prod(raw_x_xor_arbiter_out, -1)
        if self.output_type == 'y_abs_raw_x_raw':
            inverted_x_output = - raw_x_out
        elif self.output_type == 'y_abs_raw_x_abs_raw':
            inverted_x_output = torch.abs(raw_x_out)
        else:
            sigmoid_x_output = torch.sigmoid(raw_x_out)
            inverted_x_output = 1 - 2 * sigmoid_x_output
            inverted_x_output = self.x_batch_norm(inverted_x_output.unsqueeze(1)).squeeze()
        # Add x-PUF output to input feature matrix for the input of the y-PUF
        # y_feature_vector_matrix = torch.cat([feature_vector_matrix, sigmoid_x_output.unsqueeze(1)], -1)
        # Hoping that buffering the tensor is faster than creating a new one every iteration
        y_feature_vector_matrix_shape = [x.shape[0], x.shape[1] + 2]
        if self.y_feature_vector_matrix is None:
            y_feature_vector_matrix = torch.empty(y_feature_vector_matrix_shape, dtype=x.dtype)
        else:
            y_feature_vector_matrix = self.y_feature_vector_matrix
        # y_feature_vector_matrix[:, self.y_pivot:-1] *= sigmoid_x_out.unsqueeze(1)


            # Move feature vectors with higher index than pivot to make space for new bit
        y_feature_vector_matrix[:, (self.y_pivot + 1):-1] = feature_vector_matrix[:, self.y_pivot:] * torch.abs(
                inverted_x_output).unsqueeze(1)
        # Transform new challenge bit to feature vector and place it at the pivot position
        y_feature_vector_matrix[:, self.y_pivot] = inverted_x_output * feature_vector_matrix[:, self.y_pivot]
        # Add new challenge bit to all feature vectors with smaller index than the pivot
        y_feature_vector_matrix[:, :self.y_pivot] = feature_vector_matrix[:, :self.y_pivot] * inverted_x_output.unsqueeze(1)

        # Add bias term
        y_feature_vector_matrix[:, -1] = torch.abs(inverted_x_output)

        # batch_mean = y_feature_vector_matrix[:, 0].mean()
        # batch_var = y_feature_vector_matrix[:, 0].var()
        # batch_std = y_feature_vector_matrix[:, 0].std()

        # y_feature_vector_matrix = (y_feature_vector_matrix - batch_mean) / batch_std

        # self.per_batch_mean.append(batch_mean.data.numpy())
        # self.per_batch_var.append(batch_var.data.numpy())

        # correct_model = InterposePuf(**{'num_stages': 64,
        #                                 'num_x_xor_pufs': 1,
        #                                 'num_y_xor_pufs': 2,
        #                                 'create_feature_vectors': False,
        #                                 'y_pivot': 16})
        #
        # correct_model.import_weights(self.export_weights())
        #
        # y_response, x_response, y_challenge = correct_model.compute_response(x.data.numpy(), debug_output=True)
        # model_y_feature_vector = PufUtil.challenge_to_feature_vector(y_challenge)
        # model_x_feature_vector = PufUtil.challenge_to_feature_vector(x.data.numpy())

        # y_feature_vector_matrix = y_feature_vector_matrix / sigmoid_x_out.var()

        # if self.debug_counter%10 == 0:
        #     print('mean {:f}, var {:f}'.format(

        # Compute y-PUF output
        raw_y_arbiter_out = [lin_layer(y_feature_vector_matrix) for lin_layer in self.y_arbiter_layer]
        raw_y_xor_arbiter_out = torch.cat(raw_y_arbiter_out, dim=-1)
        raw_y_out = torch.prod(raw_y_xor_arbiter_out, -1)

        #net_out = (sigmoid_y_out.data.numpy() > 0.5).astype('int')
        if self.output_type == 'raw':
            output = raw_y_out
        elif self.output_type == 'abs_raw' or self.output_type == 'y_abs_raw_x_raw' or self.output_type == 'y_abs_raw_x_abs_raw':
            output = torch.abs(raw_y_out)
        elif self.output_type == 'sigmoid':
            output = torch.sigmoid(raw_y_out)
        else:
            raise ValueError()
        return output

    def export_tensor_weights(self):

        x_weights = []
        x_bias = []
        for x_layer in self.x_arbiter_layer:
            x_weights.append(x_layer.weight[0])
            x_bias.append(x_layer.bias)

        y_weights = []
        y_bias = []
        for y_layer in self.y_arbiter_layer:
            y_weights.append(y_layer.weight[0, :-1])
            y_bias.append(y_layer.weight[0, -1])
            #y_bias.append(y_layer.bias.data.numpy())

        result = {'x_weights': torch.stack(x_weights, 0),
                  'x_bias': torch.stack(x_bias, 0),
                  'y_weights': torch.stack(y_weights, 0),
                  'y_bias': torch.stack(y_bias, 0)}
        return result

    def export_weights(self):
        x_weights = []
        x_bias = []
        for x_layer in self.x_arbiter_layer:
            #x_weights.append(x_layer.weight.data.numpy().squeeze())
            x_weights.append(x_layer.weight.cpu().detach().numpy().squeeze())
            #x_bias.append(x_layer.bias.data.numpy())
            x_bias.append(x_layer.bias.cpu().detach().numpy())

        y_weights = []
        y_bias = []
        for y_layer in self.y_arbiter_layer:
            #y_weights.append(y_layer.weight.data.numpy().squeeze()[:-1])
            y_weights.append(y_layer.weight.cpu().detach().numpy().squeeze()[:-1])
            #y_bias.append(y_layer.weight.data.numpy().squeeze()[-1])
            y_bias.append(y_layer.weight.cpu().detach().numpy().squeeze()[-1])
            # y_bias.append(y_layer.bias.data.numpy())

        result = {'x_weights': np.stack(x_weights, 0),
                  'x_bias': np.stack(x_bias, 0),
                  'y_weights': np.stack(y_weights, 0),
                  'y_bias': np.stack(y_bias, 0)}
        return result

    def import_weights(self, weight_dict):

        x_weights = weight_dict['x_weights']
        x_bias = weight_dict['x_bias']

        for puf_index in range(x_weights.shape[0]):
            self.x_arbiter_layer[puf_index].weight.data = torch.Tensor(x_weights[puf_index]).unsqueeze(0)
            self.x_arbiter_layer[puf_index].bias.data = torch.Tensor(x_bias[puf_index])

        y_weights = weight_dict['y_weights']
        y_bias = weight_dict['y_bias']

        for puf_index in range(y_weights.shape[0]):
            self.y_arbiter_layer[puf_index].weight.data = torch.Tensor(np.concatenate((y_weights[puf_index], y_bias[puf_index]))).unsqueeze(0)
            #self.y_arbiter_layer[puf_index].weight.data = torch.Tensor(y_weights[puf_index]).unsqueeze(0)
            #self.y_arbiter_layer[puf_index].bias.data = torch.Tensor(y_bias[puf_index])

class InterposePufNetFrozenY(InterposePufNet):

    def __init__(self, num_x_xors, num_y_xors, num_stages, y_pivot, output_type='sigmoid', input_is_feature_vector=False):
        super().__init__(num_x_xors, num_y_xors, num_stages, y_pivot, output_type, input_is_feature_vector)
        self.y_arbiter_layer.require_grad = False



class SimpleDenseNet(nn.Module):

    def __init__(self, num_inputs, num_hidden_layers, num_neurons):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = num_neurons

        layers = [[nn.Linear(num_inputs, num_neurons, bias=True), nn.ReLU()]]
        layers += [[nn.Linear(num_neurons, num_neurons, bias=True), nn.ReLU()] for _ in range(1, num_hidden_layers)]
        layers += [[nn.Linear(num_neurons, 1, bias=True), nn.Sigmoid()]]

        layers = itertools.chain.from_iterable(layers)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    # def get_forward_activations(self, x):
    #     tmp = self.net(x)
    #     print()
    #
    # def weights_to_flat_vector(self):
    #     """
    #     The goal is to concat all layer weights to a single numpy array. At the same time, some information
    #     is stored that allows the reverse process, that is setting the layer w
    #     :return:
    #     """
    #     pass


class DenseNet(nn.Module):

    def __init__(self, num_inputs, neurons_per_layer):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hidden_layers = len(neurons_per_layer)
        self.num_neurons_per_layer = neurons_per_layer

        layers = []
        last_num_neurons = 0
        for index, num_neurons in enumerate(neurons_per_layer):
            if index == 0:
                layers.append([nn.Linear(num_inputs, num_neurons, bias=True), nn.ReLU()])
            else:
                layers.append([nn.Linear(last_num_neurons, num_neurons, bias=True), nn.ReLU()])
            last_num_neurons = num_neurons
        layers += [[nn.Linear(num_neurons, 1, bias=True), nn.Sigmoid()]]

        layers = itertools.chain.from_iterable(layers)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)




class MultiInterposePufNet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.num_multi_pufs = kwargs['num_multi_pufs']
        self.num_x_xors = kwargs['num_x_xors']
        self.num_y_xors = kwargs['num_y_xors']
        self.num_stages = kwargs['num_stages']
        self.y_pivot = kwargs['y_pivot']
        self.include_x_batch_norm = kwargs.get('include_x_batch_norm', False)
        self.include_y_pseudo_norm = kwargs.get('include_y_pseudo_norm', False)

        self.x_arbiter_layer = nn.Linear(self.num_stages, self.num_multi_pufs*self.num_x_xors, bias=True)
        self.y_arbiter_layer = nn.ModuleList([nn.Linear(self.num_stages + 2, self.num_y_xors, bias=False) for _ in range(self.num_multi_pufs)])
        if self.include_x_batch_norm:
            self.x_batch_norm = nn.BatchNorm1d(self.num_multi_pufs)
        else:
            self.x_batch_norm = None

        self.debug_counter = 0
        self.per_batch_mean = []
        self.per_batch_var = []

        #self.output_type = kwargs.get('output_type', 'sigmoid')
        self.x_output_type = kwargs.get('x_output_type', 'sigmoid')
        self.y_output_type = kwargs.get('y_output_type', 'sigmoid')
        self.input_is_feature_vector = kwargs.get('input_is_feature_vector', False)

        self.y_feature_vector_matrix = None

        initial_weights = kwargs.get('initial_weights', None)
        if initial_weights is not None:
            self.import_weights(initial_weights)

        self.freeze_x = kwargs.get('freeze_x', False)
        self.freeze_y = kwargs.get('freeze_y', False)
        self.freeze_x_y(self.freeze_x, self.freeze_y)
        self.x_inversion_tensor = torch.tensor(()).new_ones(size=(1, self.num_stages))
        self.x_inversion_tensor[0, :self.y_pivot] *= -1
        self.y_inversion_tensor = torch.tensor(()).new_ones(size=(1, self.num_stages + 2))
        self.y_inversion_tensor[0, :self.y_pivot] *= -1

        if self.include_y_pseudo_norm:
            self.y_pseudo_norm = torch.empty(self.num_multi_pufs, 2, self.num_y_xors, requires_grad=True)
            self.y_pseudo_norm.data = torch.empty(self.num_multi_pufs, 2, self.num_y_xors).normal_(0, 1).data
        else:
            self.y_pseudo_norm = None

    def forward(self, x):

        if not self.input_is_feature_vector:
            # Convert input challenge bits ([0,1]) to feature vector bits [-1, 1]
            feature_vectors = []
            for current_index in range(x.shape[1]):
                feature_vectors.append(torch.prod(1 - 2 * x[:, current_index:], axis=-1))
            feature_vector_matrix = torch.stack(feature_vectors, 1)
        else:
            feature_vector_matrix = x

        # Compute x-PUF output
        raw_x_arbiter_out = self.x_arbiter_layer(feature_vector_matrix)
        raw_x_arbiter_out = raw_x_arbiter_out.reshape([-1, self.num_multi_pufs, self.num_x_xors])
        #raw_x_xor_arbiter_out = torch.cat(raw_x_arbiter_out, dim=-1)
        raw_x_out = torch.prod(raw_x_arbiter_out, -1)
        if self.x_output_type == 'raw':
            x_output = - raw_x_out
        elif self.x_output_type == 'abs_raw':
            x_output = torch.abs(raw_x_out)
            if self.include_x_batch_norm:
                x_output = self.x_batch_norm(x_output.unsqueeze(-1)).squeeze(dim=-1)
        elif self.x_output_type == 'sigmoid':
            sigmoid_x_output = torch.sigmoid(raw_x_out)
            x_output = 1 - 2 * sigmoid_x_output
            #inverted_x_output = self.x_batch_norm(inverted_x_output.unsqueeze(1)).squeeze(dim=-1)
            if self.include_x_batch_norm:
                x_output = self.x_batch_norm(x_output.unsqueeze(-1)).squeeze(dim=-1)
        elif self.x_output_type == 'sigmoid2':
            sigmoid_x_output = torch.sigmoid(raw_x_out)
            x_output = 2*sigmoid_x_output - 1
            #inverted_x_output = self.x_batch_norm(inverted_x_output.unsqueeze(1)).squeeze(dim=-1)
            if self.include_x_batch_norm:
                x_output = self.x_batch_norm(x_output.unsqueeze(-1)).squeeze(dim=-1)
        elif self.x_output_type == 'hard_decision':
            if self.x_arbiter_layer.weight.requires_grad:
                #raise Exception('Trying to perform hard decision for x layer even though gradients are computed for it')
                pass
            x_output = torch.sign(raw_x_out)
        elif self.x_output_type == 'relu':
            x_output = torch.relu(raw_x_out) - 1
        else:
            raise ValueError
        # Add x-PUF output to input feature matrix for the input of the y-PUF
        # y_feature_vector_matrix = torch.cat([feature_vector_matrix, sigmoid_x_output.unsqueeze(1)], -1)
        # Hoping that buffering the tensor is faster than creating a new one every iteration
        # TODO: THIS 'BUFFERING' DOES NOT WORK. self.y_feature_vector_matrix is never set
        y_feature_vector_matrix_shape = [x.shape[0], self.num_stages + 2]
        if self.y_feature_vector_matrix is None:
            y_feature_vector_matrix = [torch.empty(y_feature_vector_matrix_shape, dtype=x.dtype) for tmp in range(self.num_multi_pufs)]
        else:
            y_feature_vector_matrix = self.y_feature_vector_matrix
        # y_feature_vector_matrix[:, self.y_pivot:-1] *= sigmoid_x_out.unsqueeze(1)

        raw_y_out_list = []
        raw_y_arbiter_out_list = []
        raw_inverted_y_arbiter_out_list = []

        for current_mult_puf in range(self.num_multi_pufs):
            # Move feature vectors with higher index than pivot to make space for new bit
            y_feature_vector_matrix[current_mult_puf][:, (self.y_pivot + 1):-1] = feature_vector_matrix[:, self.y_pivot:] * torch.abs(
                    x_output[:, current_mult_puf:current_mult_puf+1])
            # Transform new challenge bit to feature vector and place it at the pivot position
            y_feature_vector_matrix[current_mult_puf][:, self.y_pivot] = x_output[:, current_mult_puf] * feature_vector_matrix[:, self.y_pivot]
            # Add new challenge bit to all feature vectors with smaller index than the pivot
            y_feature_vector_matrix[current_mult_puf][:, :self.y_pivot] = feature_vector_matrix[:, :self.y_pivot] * x_output[:, current_mult_puf:current_mult_puf+1]

            # Add bias term
            y_feature_vector_matrix[current_mult_puf][:, -1] = torch.abs(x_output[:, current_mult_puf])

            if self.include_y_pseudo_norm:
                raw_y_arbiter_first = self.y_arbiter_layer[current_mult_puf].weight.data[:,:self.y_pivot+1] @ y_feature_vector_matrix[current_mult_puf][:, :self.y_pivot+1].T
                raw_y_arbiter_second = self.y_arbiter_layer[current_mult_puf].weight.data[:,self.y_pivot+1:] @ y_feature_vector_matrix[current_mult_puf][:, self.y_pivot+1:].T
                raw_y_arbiter_first *= self.y_pseudo_norm[current_mult_puf,0,:].unsqueeze(1)
                raw_y_arbiter_second *= self.y_pseudo_norm[current_mult_puf,1,:].unsqueeze(1)
                raw_y_arbiter_out = (raw_y_arbiter_first+raw_y_arbiter_second).T
            else:
                # Compute y-PUF output
                raw_y_arbiter_out = self.y_arbiter_layer[current_mult_puf](y_feature_vector_matrix[current_mult_puf])
            raw_y_out = torch.prod(raw_y_arbiter_out, -1)
            raw_y_out_list.append(raw_y_out)
            raw_y_arbiter_out_list.append(raw_y_arbiter_out)

            if self.y_output_type == 'sigmoid_and_abs_raw':
                inverted_raw_y_out = y_feature_vector_matrix[current_mult_puf] @ (self.y_arbiter_layer[current_mult_puf].weight * self.y_inversion_tensor).T
                raw_inverted_y_arbiter_out_list.append(inverted_raw_y_out)
        multi_raw_y_out = torch.stack(raw_y_out_list)


        #net_out = (sigmoid_y_out.data.numpy() > 0.5).astype('int')
        if self.y_output_type == 'raw':
            output = multi_raw_y_out
        elif self.y_output_type == 'abs_raw':
            output = torch.abs(multi_raw_y_out)
        elif self.y_output_type == 'sigmoid':
            output = torch.sigmoid(multi_raw_y_out)
        elif self.y_output_type == 'sigmoid_and_abs_raw':
            x_raw_out = raw_x_arbiter_out.transpose(0,1).transpose(1,2) # dimensions are now [num_multi, num_xor, batch_size]
            x_raw_out = torch.abs(x_raw_out)
            multi_raw_y_arbiter_out = torch.stack(raw_y_arbiter_out_list)
            inverted_multi_raw_y_arbiter_out = torch.stack(raw_y_arbiter_out_list)
            y_raw_out = torch.abs(multi_raw_y_arbiter_out).transpose(1,2)
            inverted_y_raw_out = torch.abs(inverted_multi_raw_y_arbiter_out).transpose(1,2)

            inverted_raw_x_arbiter_out = feature_vector_matrix @ (self.x_inversion_tensor*self.x_arbiter_layer.weight).transpose(0,1) + self.x_arbiter_layer.bias
            inverted_raw_x_arbiter_out = inverted_raw_x_arbiter_out.reshape([-1, self.num_multi_pufs, self.num_x_xors])
            inverted_raw_x_arbiter_out = inverted_raw_x_arbiter_out.transpose(0,1).transpose(1,2)
            output = (torch.sigmoid(multi_raw_y_out), x_raw_out, inverted_raw_x_arbiter_out, y_raw_out, inverted_y_raw_out)
        else:
            raise ValueError()
        return output

    def export_tensor_weights(self):

        x_weights = self.x_arbiter_layer.weight.reshape(self.num_multi_pufs, self.num_x_xors, -1)
        x_bias = self.x_arbiter_layer.weight.reshape(self.num_multi_pufs, self.num_x_xors, -1)

        y_weights = []
        y_bias = []
        for y_layer in self.y_arbiter_layer:
            y_weights.append(y_layer.weight[:, :-1])
            y_bias.append(y_layer.weight[:, -1])
            #y_bias.append(y_layer.bias.data.numpy())

        result = {'x_weights': x_weights,
                  'x_bias': x_bias,
                  'y_weights': torch.stack(y_weights, 0),
                  'y_bias': torch.stack(y_bias, 0)}
        if self.include_y_pseudo_norm:
            result.update(dict(y_pseudo_norm=self.y_pseudo_norm.data))
        return result

    def export_weights(self, export_single_multi_puf=-1):
        x_weights = self.x_arbiter_layer.weight.cpu().detach().numpy().reshape(self.num_multi_pufs, self.num_x_xors, -1)
            #x_bias.append(x_layer.bias.data.numpy()
        x_bias = self.x_arbiter_layer.bias.cpu().detach().numpy().reshape(self.num_multi_pufs, self.num_x_xors, -1)

        y_weights = []
        y_bias = []
        for y_layer in self.y_arbiter_layer:
            #y_weights.append(y_layer.weight.data.numpy().squeeze()[:-1])
            y_weights.append(y_layer.weight.cpu().detach().numpy()[:, :-1])
            #y_bias.append(y_layer.weight.data.numpy().squeeze()[-1])
            y_bias.append(y_layer.weight.cpu().detach().numpy()[:, -1])
            # y_bias.append(y_layer.bias.data.numpy())

        y_weights = np.stack(y_weights, 0)
        y_bias = np.stack(y_bias, 0)

        if export_single_multi_puf >= 0:
            x_weights = x_weights[export_single_multi_puf]
            x_bias = x_bias[export_single_multi_puf]
            y_weights = y_weights[export_single_multi_puf]
            y_bias = y_bias[export_single_multi_puf]
        result = {'x_weights': x_weights,
                  'x_bias': x_bias,
                  'y_weights': y_weights,
                  'y_bias': y_bias}
        if self.include_y_pseudo_norm:
            result.update(dict(y_pseudo_norm=self.y_pseudo_norm.data.cpu().detach().numpy()))
        return result

    def import_weights(self, weight_dict, import_from_single_ipuf, invert_one_y_arbiter=False, invert_one_x_arbiter=False):

        if 'x_weights' in weight_dict.keys() and 'x_bias' in weight_dict.keys():
            x_weights = weight_dict['x_weights']
            x_bias = weight_dict['x_bias']

            if import_from_single_ipuf:
                x_weights = np.tile(x_weights, (self.num_multi_pufs,1,1))
                #x_bias = np.tile(x_bias, (self.num_multi_pufs))
                x_bias = np.array(x_bias.flatten().tolist()*self.num_multi_pufs)

            if invert_one_x_arbiter:
                x_weights[:,0, :] *= -1
                x_bias[::self.num_x_xors] *= -1

            self.x_arbiter_layer.weight.data = torch.Tensor(x_weights.reshape(self.num_multi_pufs*self.num_x_xors, -1))
            self.x_arbiter_layer.bias.data = torch.Tensor(x_bias.flatten().squeeze())

        if 'y_weights' in weight_dict.keys() and 'y_bias' in weight_dict.keys():

            y_weights = weight_dict['y_weights']
            y_bias = weight_dict['y_bias']

            if import_from_single_ipuf:
                y_weights = np.tile(y_weights, (self.num_multi_pufs,1,1))
                y_bias = np.tile(y_bias, (self.num_multi_pufs, 1, 1))

            if len(y_bias.shape) < 3:
                y_bias = y_bias[:,:,np.newaxis]

            for puf_index in range(self.num_multi_pufs):
                self.y_arbiter_layer[puf_index].weight.data = torch.Tensor(np.concatenate([y_weights[puf_index], y_bias[puf_index]], axis=1))
                #self.y_arbiter_layer[puf_index].weight.data = torch.Tensor(y_weights[puf_index]).unsqueeze(0)
                #self.y_arbiter_layer[puf_index].bias.data = torch.Tensor(y_bias[puf_index])

                if invert_one_y_arbiter:
                    self.y_arbiter_layer[puf_index].weight.data[0,:] *= -1


    def freeze_x_y(self, freeze_x, freeze_y):
        if freeze_x:
            self.x_arbiter_layer.requires_grad = False
        if freeze_y:
            for current_y in range(self.num_multi_pufs):
                self.y_arbiter_layer[current_y].requires_grad = False




class MultiSplitXorArbiterNet(nn.Module):

    def __init__(self, num_multi_pufs, num_stages, split_index, output_type='sigmoid', input_is_feature_vector=True):
        super().__init__()
        self.num_multi_pufs = num_multi_pufs
        self.num_stages = num_stages
        self.split_index = split_index

        # self.left_arbiter_layer = nn.ModuleList([nn.Linear(split_index, 1, bias=False) for _ in range(num_xors)])
        # self.right_arbiter_layer = nn.ModuleList([nn.Linear(num_stages-split_index, 1, bias=True) for _ in range(num_xors)])
        self.left_arbiter_layer = nn.Linear(split_index, num_multi_pufs, bias=False)
        self.right_arbiter_layer = nn.Linear(num_stages-split_index, num_multi_pufs, bias=True)
        self.left_arbiter_batch_norm = nn.BatchNorm1d(num_multi_pufs)
        self.right_arbiter_batch_norm = nn.BatchNorm1d(num_multi_pufs)

        self.output_type = output_type

    def forward(self, x):
        left_raw_arbiter_out = self.left_arbiter_layer(x[:, :self.split_index])
        right_raw_arbiter_out = self.right_arbiter_layer(x[:, self.split_index:])
        # left_normed_arbiter_out = self.left_arbiter_batch_norm(left_raw_arbiter_out)
        # right_normed_arbiter_out = self.right_arbiter_batch_norm(right_raw_arbiter_out)
        left_normed_arbiter_out = left_raw_arbiter_out
        right_normed_arbiter_out = right_raw_arbiter_out
        raw_out = right_normed_arbiter_out**2 - left_normed_arbiter_out**2

        if self.output_type == 'raw':
            output = raw_out
        # elif self.output_type == 'abs_raw':
        #     output = torch.abs(raw_out)
        # elif self.output_type == 'sigmoid':
        #     output = torch.sigmoid(raw_out)
        else:
            raise ValueError()
        # make sure that dimensions are [num_multi_pufs, num_samples]
        return output.transpose(0, 1)

    def export_weights(self):
        # weights = []
        # bias = []
        # for layer in self.arbiter_layer:
        #     weights.append(layer.weight.data.numpy().squeeze())
        #     bias.append(layer.bias.data.numpy())

        result = {'left_weights': self.left_arbiter_layer.weight.detach().numpy(),
                  'right_weights': self.right_arbiter_layer.weight.detach().numpy(),
                  'right_bias': self.right_arbiter_layer.bias.detach().numpy()}
        return result

    def export_tensor_weights(self):
        result = {'left_weights': self.left_arbiter_layer.weight,
                  'right_weights': self.right_arbiter_layer.weight,
                  'right_bias': self.right_arbiter_layer.bias}
        return result



class MultiHybridNeuralNetInterposePufNet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.num_multi_pufs = kwargs.get('num_multi_pufs')
        self.num_x_xors = kwargs.get('num_x_xors')
        self.num_y_xors = kwargs.get('num_y_xors')
        self.num_stages = kwargs.get('num_stages')
        self.y_pivot = kwargs.get('y_pivot')


        self.x_num_neurons_per_layer = kwargs.get('x_neurons_per_layer')
        self.x_num_hidden_layers = len(self.x_num_neurons_per_layer)

        x_multi_nets = []
        for current_multipuf in range(self.num_multi_pufs):
            layers = []
            last_num_neurons = 0
            for index, num_neurons in enumerate(self.x_num_neurons_per_layer):
                if index == 0:
                    layers.append([nn.Linear(self.num_stages, num_neurons, bias=True), nn.ReLU()])
                else:
                    layers.append([nn.Linear(last_num_neurons, num_neurons, bias=True), nn.ReLU()])
                last_num_neurons = num_neurons
            layers += [[nn.Linear(num_neurons, 1, bias=True)]]

            layers = itertools.chain.from_iterable(layers)
            x_multi_nets.append(nn.Sequential(*layers))
        self.x_multi_nets = nn.ModuleList(x_multi_nets)

        self.y_arbiter_layer = nn.ModuleList([nn.Linear(self.num_stages + 2, self.num_y_xors, bias=False) for _ in range(self.num_multi_pufs)])

        self.debug_counter = 0
        self.per_batch_mean = []
        self.per_batch_var = []

        self.x_output_type = kwargs.get('x_output_type', 'sigmoid')
        self.y_output_type = kwargs.get('y_output_type', 'sigmoid')
        self.input_is_feature_vector = kwargs.get('input_is_feature_vector', False)

        self.y_feature_vector_matrix = None

        initial_weights = kwargs.get('initial_weights', None)
        if initial_weights is not None:
            self.import_weights(initial_weights)

        self.freeze_x = kwargs.get('freeze_x', False)
        self.freeze_y = kwargs.get('freeze_y', False)
        self.freeze_x_y(self.freeze_x, self.freeze_y)

        self.output_x_only = False

    def forward(self, x):

        if not self.input_is_feature_vector:
            # Convert input challenge bits ([0,1]) to feature vector bits [-1, 1]
            feature_vectors = []
            for current_index in range(x.shape[1]):
                feature_vectors.append(torch.prod(1 - 2 * x[:, current_index:], axis=-1))
            feature_vector_matrix = torch.stack(feature_vectors, 1)
        else:
            feature_vector_matrix = x

        # Compute x-PUF output
        multi_x_out = torch.stack([tmp(feature_vector_matrix) for tmp in self.x_multi_nets], axis=0)
        #raw_x_xor_arbiter_out = torch.cat(raw_x_arbiter_out, dim=-1)
        if self.x_output_type == 'raw':
            x_output = multi_x_out
        elif self.x_output_type == 'abs_raw':
            x_output = torch.abs(multi_x_out)
            if self.include_x_batch_norm:
                x_output = self.x_batch_norm(x_output.unsqueeze(-1)).squeeze(dim=-1)
        elif self.x_output_type == 'sigmoid':
            x_output = torch.sigmoid(multi_x_out)
        elif self.x_output_type == 'tanh':
            sigmoid_x_output = torch.sigmoid(multi_x_out)
            x_output = 1 - 2 * sigmoid_x_output
            #inverted_x_output = self.x_batch_norm(inverted_x_output.unsqueeze(1)).squeeze(dim=-1)
        elif self.x_output_type == 'hard_decision':
            if self.x_arbiter_layer.requires_grad:
                raise Exception('Trying to perform hard decision for x layer even though gradients are computed for it')
            x_output = torch.sign(multi_x_out)
        elif self.x_output_type == 'relu':
            x_output = torch.relu(multi_x_out) - 1
        else:
            raise ValueError
        # Add x-PUF output to input feature matrix for the input of the y-PUF
        # y_feature_vector_matrix = torch.cat([feature_vector_matrix, sigmoid_x_output.unsqueeze(1)], -1)
        # Hoping that buffering the tensor is faster than creating a new one every iteration
        y_feature_vector_matrix_shape = [x.shape[0], self.num_stages + 2]
        if self.y_feature_vector_matrix is None:
            y_feature_vector_matrix = [torch.empty(y_feature_vector_matrix_shape, dtype=x.dtype) for tmp in range(self.num_multi_pufs)]
        else:
            y_feature_vector_matrix = self.y_feature_vector_matrix
        # y_feature_vector_matrix[:, self.y_pivot:-1] *= sigmoid_x_out.unsqueeze(1)

        if not self.output_x_only:
            raw_y_out_list = []
            for current_mult_puf in range(self.num_multi_pufs):
                # Move feature vectors with higher index than pivot to make space for new bit
                y_feature_vector_matrix[current_mult_puf][:, (self.y_pivot + 1):-1] = feature_vector_matrix[:, self.y_pivot:] * torch.abs(
                        x_output[:, current_mult_puf:current_mult_puf+1])
                # Transform new challenge bit to feature vector and place it at the pivot position
                y_feature_vector_matrix[current_mult_puf][:, self.y_pivot] = x_output[:, current_mult_puf] * feature_vector_matrix[:, self.y_pivot]
                # Add new challenge bit to all feature vectors with smaller index than the pivot
                y_feature_vector_matrix[current_mult_puf][:, :self.y_pivot] = feature_vector_matrix[:, :self.y_pivot] * x_output[:, current_mult_puf:current_mult_puf+1]

                # Add bias term
                y_feature_vector_matrix[current_mult_puf][:, -1] = torch.abs(x_output[:, current_mult_puf])

                # Compute y-PUF output
                raw_y_arbiter_out = self.y_arbiter_layer[current_mult_puf](y_feature_vector_matrix[current_mult_puf])
                raw_y_out = torch.prod(raw_y_arbiter_out, -1)
                raw_y_out_list.append(raw_y_out)

            multi_raw_y_out = torch.stack(raw_y_out_list)

            #net_out = (sigmoid_y_out.data.numpy() > 0.5).astype('int')
            if self.y_output_type == 'raw':
                output = multi_raw_y_out
            elif self.y_output_type == 'abs_raw':
                output = torch.abs(multi_raw_y_out)
            elif self.y_output_type == 'sigmoid':
                output = torch.sigmoid(multi_raw_y_out)
            else:
                raise ValueError()

        else:
            output = x_output
        return output

    def export_tensor_weights(self):

        y_weights = []
        y_bias = []
        for y_layer in self.y_arbiter_layer:
            y_weights.append(y_layer.weight[:, :-1])
            y_bias.append(y_layer.weight[:, -1])
            #y_bias.append(y_layer.bias.data.numpy())

        result = { 'y_weights': torch.stack(y_weights, 0),
                  'y_bias': torch.stack(y_bias, 0)}
        return result

    def export_weights(self):
        y_weights = []
        y_bias = []
        for y_layer in self.y_arbiter_layer:
            #y_weights.append(y_layer.weight.data.numpy().squeeze()[:-1])
            y_weights.append(y_layer.weight.cpu().detach().numpy()[:, :-1])
            #y_bias.append(y_layer.weight.data.numpy().squeeze()[-1])
            y_bias.append(y_layer.weight.cpu().detach().numpy()[:, -1])
            # y_bias.append(y_layer.bias.data.numpy())

        result = { 'y_weights': np.stack(y_weights, 0),
                  'y_bias': np.stack(y_bias, 0)}
        return result

    def import_weights(self, weight_dict, import_from_single_ipuf):

        if 'y_weights' in weight_dict.keys() and 'y_bias' in weight_dict.keys():

            y_weights = weight_dict['y_weights']
            y_bias = weight_dict['y_bias']

            if import_from_single_ipuf:
                y_weights = np.tile(y_weights, (self.num_multi_pufs,1,1))
                y_bias = np.tile(y_bias, (self.num_multi_pufs, 1, 1))

            for puf_index in range(self.num_multi_pufs):
                self.y_arbiter_layer[puf_index].weight.data = torch.Tensor(np.concatenate([y_weights[puf_index], y_bias[puf_index]], axis=1))
                #self.y_arbiter_layer[puf_index].weight.data = torch.Tensor(y_weights[puf_index]).unsqueeze(0)
                #self.y_arbiter_layer[puf_index].bias.data = torch.Tensor(y_bias[puf_index])


    def freeze_x_y(self, freeze_x, freeze_y):
        if freeze_x:
            self.x_arbiter_layer.requires_grad = False
        if freeze_y:
            for current_y in range(self.num_multi_pufs):
                self.y_arbiter_layer[current_y].requires_grad = False