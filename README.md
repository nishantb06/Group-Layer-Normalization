# Implemantation of Group Normalization and Layer Normalization
# This code is all about
1. Including  group and layer normalisation instead of simple batch normalisations.
2. Find 10 images that were misclassified by each of the 3 models
    1. Group Normalisation
    2. Layer Normalisation
    2. Batch Normalisation + L1 regularization
3. Plot the following 4 curves train accuracy , test accuracy ,test loss, train loss for each of the 3 models.

# File structure 
1. **utils.py** - Contains helper function to get the correct instance of normalisation according to the input  passed
2. **models.py** - CNN model architecture with less than 10,000 parameters with a capability to get 99.4 % accuracy consistently in less than 15 epochs 
3. **dataloader.py** : Contains code for the train and test data loaders for the MNIST digits. Various augmentation can be added here as well.
4. **train.py** : training function
5. **test.py** : test function 
6. **Group_Layer_Normalisation.ipynb** : Colab notebook for training on GPU's which imports necessaay classes/functions from above files. all results can be seen here

# What is [Group Normalisation](https://arxiv.org/pdf/1803.08494v3.pdf) and [Layer Normalisation](https://arxiv.org/abs/1607.06450)

Smaller batch sizes are not good representatives of the entire population. Whnever we need to work with smaller batch sizes or train across multiple GPU's its always better to normalise within the datapoint(an image/channel in our case) rather than normalising multiple inputs. Layer and Group Normalisation are techniques to do **Normalise within the datapoint**.

![Imgur](https://imgur.com/PeERSaz.png)

## Batch Normalisation

Refering to the above image, it is a techinique to normalise data across various data points. Therfore all channels of the same index are grouped together and normalised (4 purple boxes).
We will get 4 mean and std deviation  pairs as well.

mean = [0.5,0.6,0.1,0.2,0.4,0.2,0.7,0.1,0.1,0.4,0.5,0.6,0.7]/12 = 0.3923

std = 0.2290

 No. of trainable parameters = 8 as there will be one gamma,beta pair for each mean,std deviation pair

## Layer Normalisation 

Refering to the above image it is a technique in which we normalise  all the values in the yellow box. **Normalisation is done across all the channels of a particular layer, for each image in the batch**. Hence if there  are 3 images in a batch we would get 3 values of mean and standard deviation respectively

For Layer 1

mean = (0.5+0.6+0.1+0.2-0.5+0.1+0.6+0.4+0.5+0.6+0.1+0.2+0.4+0.4+0.4+0.4)/16 = 0.3125

std deviation = 0.2802

Similarly we can calculate 2 more pairs of mean.std deviation. No of trainable parameters will be 6, that is 3 pairs of gamma and beta

## Group Normalisation

Refering to the above image it is a techinque in which we divide the channels of particular layer into n predifined  group and calculate mean and standard deviation of the points belonging to the paricular group only. Like for one data point we can divide the layer into two groups(Red Boxes). For the entire batch we will get 6 groups in total.

Group 1 
mean = [0.4,0.5,0.6,0.7,0.4,0.5,0.6,0.7]/8 = 0.55

std dev = 0.1195

Group 2 = Same as  group 1 because it has the same data

Number of trainable parameters will be 2 x number of mean,std deviation pairs = 2 x 6 = 12 gamma,beta pairs.

## Results
### Misclassified images 
Group Normalisations                        
![Imgur](https://imgur.com/cWqHo11.png) 

Layer  Normalisation

![Imgur](https://imgur.com/3ApybsI.png)

Batch + L1 Normalisation

![Imgur](https://imgur.com/RZ6Kt5N.png)

### Training/Testing curves
Batch Norm+L1 training/test curves

![Imgur](https://imgur.com/56mB6KA.png)

Group Norm training/testing curves

![Imgur](https://imgur.com/zZ5wMhR.png)

Layer Norm train/test curves

![Imgur](https://imgur.com/O8ylxbm.png)

# Inferences
1. Best accuracies

|  Norm | Best test accuracy | Best train accuracy|
|-------|--------------------|--------------------|
|Layer  | 99.04%             | 99.11%             |
|Group  | 99.10%             | 99.43%             |
|Batch+L1  | 99.24%             | 99.40%             |

2. Roughly all the 3 normalisations yield the same result which is expected  because the batch size is fairly large(128). On close inspection the Group normalistion model is overfitting more  than the other 2 model

3. The best model is of Batch normalisation + L1, Although L1 regularization is not as effective(we can still see some overfitting), which is expected as L1 doesn't work very well with CNN's