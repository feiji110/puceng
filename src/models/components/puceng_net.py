import torch
from torch import nn
from src.models.components.resnetblock import ResNetBlock

class PUCENGResNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 102,
        lin1_size: int = 256,
        lin2_size: int = 512,
        lin3_size: int = 1024,
        lin4_size: int = 2048,
        lin5_size: int = 1024,
        output_size: int = 1001,
        dropout_rate: float = 0.25,  # Add dropout_rate as a parameter
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        :param dropout_rate: The dropout rate to be applied.
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout layer

            ResNetBlock(lin1_size, lin2_size),
            # nn.BatchNorm1d(lin2_size),
            # nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout layer

            ResNetBlock(lin2_size, lin3_size),
             
            # nn.BatchNorm1d(lin3_size),
            # nn.ReLU(),
            nn.Dropout(dropout_rate),

            ResNetBlock(lin3_size, lin4_size),
            nn.Dropout(dropout_rate),  # Add dropout layer
            
            ResNetBlock(lin4_size, lin5_size),
            nn.Dropout(dropout_rate),  # Add dropout layer

            nn.Linear(lin5_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, features = x.size()

        x = x.view(batch_size, -1)

        return self.model(x)
