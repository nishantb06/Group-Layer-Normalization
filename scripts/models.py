import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_normalisation

# 9972 params / 9977 / 130,000 params for BN,group and layer batch norm respectively
dropout_value = 0.02

class Net4(nn.Module):
  def __init__(self,norm,batch_size = 2):
    super(Net4, self).__init__()

    self.batch_size = batch_size
    self.norm = norm
  
    # Input Block  LAYER  1
    self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
        nn.ReLU(),
        get_normalisation(self.norm,input_channels=10,num_groups=2,shape = (self.batch_size,10,28,28)),
        nn.Dropout(dropout_value)
        ) # output_size = 28 rf=3

    self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            get_normalisation(self.norm,input_channels=16,num_groups=4,shape = (self.batch_size,16,28,28)),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5
    

    #TRANSITION BLOCK 1
    self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14 rf=6
    self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            get_normalisation(self.norm,input_channels=10,num_groups=2,shape = (self.batch_size,10,14,14)),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=10

    #LAYER 2
    self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            get_normalisation(self.norm,input_channels=14,num_groups=2,shape = (self.batch_size,14,12,12)),
            nn.ReLU()
        )
    self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=24, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            get_normalisation(self.norm,input_channels=24,num_groups=4,shape = (self.batch_size,24,10,10)),
            nn.Dropout(dropout_value)
        ) # output_size = 10 rf=14


    #TRANSITION BLOCK 2
    self.pool2 = nn.MaxPool2d(2, 2) # output_size = 5 rf=15
    self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU(),
            get_normalisation(self.norm,input_channels=10,num_groups=2,shape = (self.batch_size,10,5,5)),
            nn.Dropout(dropout_value)
        ) # output_size = 28 rf=5
    
    #LAYER 3
    self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=15, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            get_normalisation(self.norm,input_channels=15,num_groups=3,shape = (self.batch_size,15,5,5))
        ) # output_size = 3 rf=18
    self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            get_normalisation(self.norm,input_channels=15,num_groups=3,shape = (self.batch_size,15,5,5))
        ) # output_size = 3 rf=18

    #GLOBAL AVG POOLINNG
    self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )# output size = 1 rf=34

    self.fc1 = nn.Linear(in_features=15, out_features=10)

  def forward(self,x):
    x = self.convblock1(x)
    x = self.convblock2(x)

    x = self.pool1(x)
    x = self.convblock3(x)
    
    x = self.convblock4(x)
    x = self.convblock5(x)

    x = self.pool2(x)
    x = self.convblock6(x)
    
    x = self.convblock7(x)
    x = self.convblock8(x)

    x = self.gap(x)
    x = x.view(-1,15)
    x = self.fc1(x)
      
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)