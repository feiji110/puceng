#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   EmbeddingLayer.py
@Time    :   2023/12/14 10:39:14
@Author  :   Hengda.Gao
@Contact :   ghd@nudt.edu.com
'''
import torch
import torch.nn as nn
#To.list https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#normalization-layers

class EmbeddingLayer(nn.Module):
    """
    Custom layer which performs nonlinear transform on embeddings

    Args:
        input_features (int): Number of input features.
        output_features (int): Number of output features.
        activation (str, optional): Activation function to be applied. 
            Defaults to 'silu'.
        normalization (str, optional): Normalization method to be applied.
            Defaults to 'batch_norm'.
    """

    def __init__(self, input_features, output_features, activation='silu', normalization='batch_norm') -> None:
        super().__init__()
        # Define a Sequential model containing Linear layer, normalization, and activation
        self.mlp = nn.Sequential(
            nn.Linear(input_features, output_features),
            self.get_normalization(normalization, output_features),
            self.get_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        return self.mlp(x)

    def get_activation(self, activation ):
        """
        Returns the activation function module based on the provided string.

        Args:
            activation (str): Activation function name.

        Returns:
            torch.nn.Module: Activation function module.
        """
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'leakyrelu':
            return nn.LeakyReLU()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'silu':
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def get_normalization(self, normalization, num_features):
        """
        Returns the normalization layer module based on the provided string.

        Args:
            normalization (str): Normalization method name.
            num_features (int): Number of features.

        Returns:
            torch.nn.Module: Normalization layer module.
        """
        if normalization.lower() == 'batch_norm':
            return nn.BatchNorm1d(num_features)
        elif normalization.lower() == 'layer_norm':
            return nn.LayerNorm(num_features)
        elif normalization.lower() == 'instance_norm':
            return nn.InstanceNorm1d(num_features)
        else:
            raise ValueError(f"Unsupported normalization method: {normalization}")
if __name__ == "__main__":
    input_features = 10
    output_features = 20
    activation_function = 'relu'
    normalization_method = 'batch_norm'
    embedding_layer = EmbeddingLayer(input_features, output_features, activation=activation_function, normalization=normalization_method)