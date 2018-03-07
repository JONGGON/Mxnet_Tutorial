import numpy as np
import mxnet as mx
import mxnet.gluon as gluon
import mxnet.ndarray as nd
import mxnet.autograd as autograd
import matplotlib.pyplot as plt
from tqdm import *
import os

''' Autoencoder '''

def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

#MNIST dataset
def MNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(root="MNIST" , train = True , transform = transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    # last_batch  "discard" : last_batch must be a discard or rollover , The batchsize of 'test_data' must be greater than "column_size x row_size".
    test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(root="MNIST", train = False , transform = transform) ,128 , shuffle=False, last_batch="discard") #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data

#MFashionNIST dataset
def FashionMNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = True , transform = transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    # last_batch  "discard" : last_batch must be a discard or rollover , The batchsize of 'test_data' must be greater than "column_size x row_size".
    test_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = False , transform = transform) ,128 , shuffle=False, last_batch="discard") #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data


#evaluate the data
def generate_image(data_iterator , network , ctx , dataset):

    for data, label in data_iterator:

        data = data.as_in_context(ctx).reshape((data.shape[0],-1))
        output = network(data)
        data = data.asnumpy() * 255.0
        output=output.asnumpy() * 255.0

    '''test'''
    column_size=10 ; row_size=10 #     column_size x row_size <= 10000

    print("show image")
    '''generator image visualization'''

    if not os.path.exists("Generate_Image"):
        os.makedirs("Generate_Image")

    if dataset=="MNIST":
        fig_g, ax_g = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_g.suptitle('MNIST_generator')
        for j in range(row_size):
            for i in range(column_size):
                ax_g[j][i].set_axis_off()
                ax_g[j][i].imshow(np.reshape(output[i + j * column_size],(28,28)),cmap='gray')
        fig_g.savefig("Generate_Image/MNIST_generator.png")

        '''real image visualization'''
        fig_r, ax_r = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_r.suptitle('MNIST_real')
        for j in range(row_size):
            for i in range(column_size):
                ax_r[j][i].set_axis_off()
                ax_r[j][i].imshow(np.reshape(data[i + j * column_size],(28,28)),cmap='gray')
        fig_r.savefig("Generate_Image/MNIST_real.png")

    elif dataset=="FashionMNIST":
        fig_g, ax_g = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_g.suptitle('FashionMNIST_generator')
        for j in range(row_size):
            for i in range(column_size):
                ax_g[j][i].set_axis_off()
                ax_g[j][i].imshow(np.reshape(output[i + j * column_size],(28,28)),cmap='gray')
        fig_g.savefig("Generate_Image/FashionMNIST_generator.png")

        '''real image visualization'''
        fig_r, ax_r = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig_r.suptitle('FashionMNIST_real')
        for j in range(row_size):
            for i in range(column_size):
                ax_r[j][i].set_axis_off()
                ax_r[j][i].imshow(np.reshape(data[i + j * column_size],(28,28)),cmap='gray')
        fig_r.savefig("Generate_Image/FashionMNIST_real.png")

    plt.show()

def Autoencoder(epoch = 100 , batch_size=128, save_period=10 , load_period=100 ,optimizer="sgd",learning_rate= 0.01 , dataset = "MNIST", ctx=mx.gpu(0)):

    #data selection
    if dataset =="MNIST":
        train_data , test_data = MNIST(batch_size)
        path = "weights/MNIST-{}.params".format(load_period)
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

    #Autoencoder
    net = gluon.nn.Sequential() # stacks 'Block's sequentially
    with net.name_scope():
        net.add(gluon.nn.Dense(units=200 , activation="sigmoid", use_bias=True)) 
        net.add(gluon.nn.Dropout(0.2))
        net.add(gluon.nn.Dense(units=100 , activation="sigmoid", use_bias=True))
        net.add(gluon.nn.Dropout(0.2))
        net.add(gluon.nn.Dense(units=100 , activation="sigmoid", use_bias=True))
        net.add(gluon.nn.Dropout(0.2))
        net.add(gluon.nn.Dense(units=200 , activation="sigmoid", use_bias=True))
        net.add(gluon.nn.Dropout(0.2))
        net.add(gluon.nn.Dense(units=784 , activation="sigmoid", use_bias=True))

    #weights initialization
    if os.path.exists(path):
        print("loading weights")
        net.load_params(filename=path , ctx=ctx) # weights load
    else:
        print("initializing weights")
        net.collect_params().initialize(mx.init.Normal(sigma=0.1),ctx=ctx) # weights initialization
        #net.initialize(mx.init.Normal(sigma=0.1),ctx=ctx) # weights initialization

    #optimizer
    trainer = gluon.Trainer(net.collect_params() , optimizer, {"learning_rate" : learning_rate})

    #learning
    for i in tqdm(range(1,epoch+1,1)):
        for data , label in train_data:

            data = data.as_in_context(ctx).reshape((batch_size,-1))
            data_ = data

            with autograd.record(train_mode=True):
                output=net(data)

                #loss definition
                loss=gluon.loss.L2Loss()(output,data_)
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

            elif dataset=="FashionMNIST":
                net.save_params("weights/FashionMNIST-{}.params".format(i))

    #show image
    generate_image(test_data , net , ctx , dataset)

    return "optimization completed"


if __name__ == "__main__":
    Autoencoder(epoch = 100 , batch_size=128, save_period=100 , load_period=100 ,optimizer="sgd",learning_rate= 0.01 , dataset = "MNIST", ctx=mx.gpu(0))
else :
    print("Imported")


