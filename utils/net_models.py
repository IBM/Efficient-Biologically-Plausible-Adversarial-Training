# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class net(nn.Module):          
     
    def __init__(self, input_size, hidden_size, num_classes, with_bias=False, activation_function=None):

      super(net, self).__init__()

      self.activation_function = activation_function

      if with_bias:
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True)
      else:
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)

      # initialize the layers using the He uniform initialization scheme
      fc1_limit = np.sqrt(6.0 / input_size)
      torch.nn.init.uniform_(self.fc1.weight, a=-fc1_limit, b=fc1_limit)
      fc2_limit = np.sqrt(6.0 / hidden_size)
      torch.nn.init.uniform_(self.fc2.weight, a=-fc2_limit, b=fc2_limit)

    
    def forward(self, x, do_masks=None):

      # apply activation function
      if self.activation_function is None:
        print('No activation function specified. Using ReLU.')
        x = F.relu(self.fc1(x.double()))
      elif self.activation_function=='relu':
        x = F.relu(self.fc1(x.double()))
      elif self.activation_function=='sigmoid':
        x = F.sigmoid(self.fc1(x.double()))
      elif self.activation_function=='tanh':
        x = F.tanh(self.fc1(x.double()))
      elif self.activation_function=='cap_relu':
        x = torch.clamp(F.relu(self.fc1(x.double())), min=None, max=1)

      # apply dropout
      if do_masks is not None:
        x = x * do_masks[0]
      x = F.softmax(self.fc2(x), dim=1)
      
      return x