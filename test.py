import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

x = torch.randn(2,3)

x = torch.cat( (x,x,x), 0 )
print(x)