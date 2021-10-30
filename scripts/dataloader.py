import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

torch.manual_seed(1)
batch_size = 128

# Train Phase transformations
train_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                      # transforms.RandomAffine(degrees=0, translate=None, scale=None, shear=[-5,5,-5,5],
                                      #                       fill=0, fillcolor=None, resample=None),
                                       transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                       # Note the difference between (0.1307) and (0.1307,)
                                       ])

# Test Phase transformations
test_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,transform=train_transforms),
    batch_size=batch_size,
    shuffle=True, #data reshuffled at every epoch,
    drop_last = True, 
    **kwargs)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=test_transforms),
    batch_size=batch_size, shuffle=True,drop_last = True, **kwargs)