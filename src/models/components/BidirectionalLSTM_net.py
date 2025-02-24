import torch
from torch import nn
from src.models.components.resnetblock import ResNetBlock

class BidirectionalLSTM(nn.Module):
 
    def __init__(self,
                nIn: int = 3,
                nHidden: int = 16,
                num_layers: int = 4,
                nOut: int = 1001,
                dropout_rate: float = 0.25 ):
    
        super(BidirectionalLSTM, self).__init__()
 
        self.rnn = nn.LSTM(input_size=nIn, hidden_size = nHidden,
                           num_layers = num_layers,
                            bidirectional=True,
                            dropout=dropout_rate)

        self.embedding = nn.Sequential(  
            ResNetBlock(34*2*nHidden, nOut),
            nn.Dropout(dropout_rate),  # Add dropout layer
        )   
 
    def forward(self, input):
        inp_size = input.size()
        # [34, b, 3] -> [34, b, 2*nHidden]
        recurrent, _ = self.rnn(input.reshape(-1,inp_size[0],3))
        T, b, h = recurrent.size()
        t_rec = recurrent.view(b, T * h) #  
 
        output = self.embedding(t_rec)  # [T * b, nOut] 64* 102
        output = output.view(b, -1)
 
        return output
