import torch.nn.functional as F
from dataloader import train_loader
from test import test


from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch,L1 = False):

    model.train() #set  the model  in training mode (which means the model knows to include thing like batchnorm and dropout)
    pbar = tqdm(train_loader)
    correct  = 0
    processed  = 0
    train_acc = []
    train_losses = []

    for batch_idx, (data, target) in enumerate(pbar):

        #shifting X,y of inputs to GPU
        data, target = data.to(device), target.to(device)

        #set gradients to zero, can be done anywhere before calling loss.backward()
        #calculate output from the model
        optimizer.zero_grad()
        output = model(data)

        #calculate loss
        loss = F.nll_loss(output, target)
        
        if L1:
            l1 = 0
            for p in model.parameters():
                l1 += p.abs().sum()
            loss += l1

        loss.backward()

        #update gradients
        optimizer.step()

        #printing training logs
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        train_losses.append(loss.item())

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Train Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

    return train_losses,train_acc

# def train_L1(model, device, train_loader, optimizer, epoch):

#     model.train() #set  the model  in training mode (which means the model knows to include thing like batchnorm and dropout)
#     pbar = tqdm(train_loader)
#     correct  = 0
#     processed  = 0

#     for batch_idx, (data, target) in enumerate(pbar):

#         #shifting X,y of inputs to GPU
#         data, target = data.to(device), target.to(device)

#         #set gradients to zero, can be done anywhere before calling loss.backward()
#         #calculate output from the model
#         optimizer.zero_grad()
#         output = model(data)

#         #calculate loss
#         loss = F.nll_loss(output, target)
#         l1 = 0
#         for p in model.parameters():
#             l1 += p.abs().sum()
#         loss += l1
#         loss.backward()

#         #update gradients
#         optimizer.step()

#         #printing training logs
#         pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#         correct += pred.eq(target.view_as(pred)).sum().item()
#         processed += len(data)
#         train_losses.append(loss.item())

#         pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Train Accuracy={100*correct/processed:0.2f}')
#         train_acc.append(100*correct/processed)

#     return train_losses,train_acc




        