import numpy as np
import mxnet as mx
import mxnet.gluon as gluon
import mxnet.ndarray as nd
import mxnet.autograd as autograd
from tqdm import *
import os

def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

#MNIST dataset
def MNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(root="MNIST" , train = True , transform = transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(root="MNIST", train = False , transform = transform) ,128 , shuffle=False) #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data

#MFashionNIST dataset
def FashionMNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = True , transform = transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = False , transform = transform) ,128 , shuffle=False) #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data

#CIFAR10 dataset
def CIFAR10(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label)
    train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(root="CIFAR10", train = True, transform=transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(root="CIFAR10", train = False, transform=transform) , 128 , shuffle=False) #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data

#evaluate the data
def evaluate_accuracy(data_iterator , net , ctx , dataset):
    acc = mx.metric.Accuracy()
    for data,label in data_iterator:

        if dataset=="CIFAR10":
            data = nd.slice_axis(data=data, axis=3, begin=0, end=1)

        data = data.as_in_context(ctx).reshape((data.shape[0],-1))#as_in_context : Returns an array on the target device with the same value as this array.
        label = label.as_in_context(ctx) #as_in_context : Returns an array on the target device with the same value as this array.
        output = net(data)
        prediction = nd.argmax(output , axis=1) # (batch,10(axis=1)
        acc.update(preds = prediction , labels = label)
    return acc.get()

def muitlclass_logistic_regression(epoch = 100 , batch_size=128, save_period=10 , load_period=100 ,optimizer="sgd",learning_rate= 0.01 , dataset = "MNIST", ctx=mx.gpu(0)):

    #data selection
    if dataset =="MNIST":
        train_data , test_data = MNIST(batch_size)
        path = "weights/MNIST-{}.params".format(load_period)
    elif dataset == "CIFAR10":
        train_data, test_data = CIFAR10(batch_size)
        path = "weights/CIFAR10-{}.params".format(load_period)
    elif dataset == "FashionMNIST":
        train_data, test_data = FashionMNIST(batch_size)
        path = "weights/FashionMNIST-{}.params".format(load_period)
    else:
        return "The dataset does not exist."

    '''Follow these steps:

    •Define network
    •Initialize parameters
    •Loop over inputs
    •Forward input through network to get output
    •Compute loss with output and label
    •Backprop gradient
    •Update parameters with gradient descent.
    '''
    #logistic regression network
    net = gluon.nn.Sequential() # stacks 'Block's sequentially
    with net.name_scope():
        net.add(gluon.nn.Dense(units=10 , activation=None , use_bias=True)) # linear activation

    # weight initialization
    if os.path.exists(path):
        print("loading weights")
        net.load_params(filename=path , ctx=ctx) # weights load
    else:
        print("initializing weights")
        net.collect_params().initialize(mx.init.Normal(sigma=1.),ctx=ctx) # weights initialization
    
    #optimizer
    trainer = gluon.Trainer(net.collect_params() , optimizer, {"learning_rate" : learning_rate})

    for i in tqdm(range(1,epoch+1,1)):
        for data , label in train_data:
            if dataset == "CIFAR10":
                data = nd.slice_axis(data= data , axis=3 , begin = 0 , end=1)
            data = data.as_in_context(ctx).reshape((batch_size,-1))
            label = label.as_in_context(ctx)

            with autograd.record(train_mode=True):
                output=net(data)

                #loss definition
                loss=gluon.loss.SoftmaxCrossEntropyLoss()(output,label)
                cost=nd.mean(loss).asscalar()
            loss.backward()
            trainer.step(batch_size,ignore_stale_grad=True)

        print(" epoch : {} , last batch cost : {}".format(i,cost))

        #weight_save
        if i % save_period==0:

            if not os.path.exists("weights"):
                os.makedirs("weights")

            print("saving weights")
            if dataset=="MNIST":
                net.save_params("weights/MNIST-{}.params".format(i))

            if dataset=="FashionMNIST":
                net.save_params("weights/FashionMNIST-{}.params".format(i))

            elif dataset=="CIFAR10":
                net.save_params("weights/CIFAR10-{}.params".format(i))

    test_accuracy = evaluate_accuracy(test_data , net , ctx , dataset)
    print("Test_acc : {}".format(test_accuracy[1]))

    return "optimization completed"


if __name__ == "__main__":
    muitlclass_logistic_regression(epoch=100, batch_size=128, save_period=10 , load_period=100 , optimizer="sgd", learning_rate=0.1, dataset="MNIST", ctx=mx.gpu(0))
else :
    print("Imported")


