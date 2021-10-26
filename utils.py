import torch.nn as nn
import matplotlib.pyplot as plt

def get_normalisation(norm = 'Batch',input_channels = 0,num_groups = 0,**kwargs):
    
    '''

    Parameters:
        norm : Can be either 'Group' or 'Layer' or 'Batch'
        input_channels : no. of channels expected as input
        num_groups : Only valid for Group Normalisation, specifies the no. of groups channels will be divided into
        kwargs : for Layer Norm pass in a tuple or list or torch.Size as shape = N,C,H,W 
    
    Output:
        Returns object of the specified Normalisation to be applied

    '''

    if norm  == 'Group':
        return nn.GroupNorm(num_groups,input_channels)
    
    elif norm == 'Layer':
        return nn.LayerNorm(kwargs['shape'])
    
    return nn.BatchNorm2d(input_channels)
    

def plot_curves(train_losses,test_losses,train_acc,test_acc):

    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc[4000:])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()