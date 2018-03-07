import numpy as np
import mxnet as mx
import mxnet.gluon as gluon
import mxnet.ndarray as nd
import mxnet.autograd as autograd
from tqdm import *
import os

def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255.0, label.astype(np.float32)

#MNIST dataset
def MNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(root="MNIST" , train = True , transform = transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(root="MNIST", train = False , transform = transform) ,128 , shuffle=False) #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data

#MFashionNIST dataset
def FashionMNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = True , transform = transform) ,batch_size, shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = False , transform = transform) , 128, shuffle=False) #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data

#CIFAR10 dataset
def CIFAR10(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label)
    train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(root="CIFAR10", train = True, transform=transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(root="CIFAR10", train = False, transform=transform) ,128 ,shuffle=False) #Loads data from a dataset and returns mini-batches of data.

    return train_data , test_data

#evaluate the data
def evaluate_accuracy(data_iterator , network , ctx):
    numerator = 0
    denominator = 0

    for data, label in data_iterator:

        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = network(data) # when test , 'Dropout rate' must be 0.0

        predictions = nd.argmax(output, axis=1) # (batch_size , num_outputs)

        predictions=predictions.asnumpy()
        label=label.asnumpy()
        numerator += sum(predictions == label)
        denominator += data.shape[0]

    return (numerator / denominator)

#method1 : CNN HybridBlock
class HybridBlockNetwork(gluon.HybridBlock):

    def __init__(self, **kwargs):

        super(HybridBlockNetwork , self).__init__(**kwargs)
        with self.name_scope():

            self.conv1 = gluon.nn.Conv2D(channels=60, kernel_size=(3, 3), strides=(1, 1), use_bias=True) # MNIST : result = ( batch size , 60 , 26 , 26) , CIFAR10 : : result = ( batch size , 60 , 30 , 30)
            self.maxpool1 = gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))  # MNIST : result = (batch size , 60 , 13 , 13) , CIFAR10 : result = (batch size , 60 , 15 , 15)
            self.conv2 = gluon.nn.Conv2D(channels=30, kernel_size=(6, 6), strides=(1, 1), use_bias=True)  # MNIST :  result = ( batch size , 30 , 8 , 8), CIFAR10 :  result = ( batch size , 30 , 10 , 10)
            self.maxpool2 = gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))  # MNIST : result = (batch size , 30 , 4 , 4) , CIFAR10 : result = (batch size , 30 , 5 , 5)
            self.fnn1 = gluon.nn.Dense(units=120 , use_bias=True, flatten=True)
            self.drop1 = gluon.nn.Dropout(0.2)
            self.fnn2 = gluon.nn.Dense(units=64 , use_bias=True)
            self.drop2 = gluon.nn.Dropout(0.2)
            self.fnn3 = gluon.nn.Dense(10, use_bias=True)

    def hybrid_forward(self , F ,  x):

        x=F.Activation(data=self.conv1(x), act_type='relu')
        x=self.maxpool1(x)
        x=F.Activation(data=self.conv2(x), act_type='relu')
        x=self.maxpool2(x)
        x=F.Activation(data=self.fnn1(x), act_type='sigmoid')
        x=self.drop1(x)
        x=F.Activation(data=self.fnn2(x), act_type='sigmoid')
        x=self.drop2(x)
        x=self.fnn3(x)
        return x

# method2 : CNN Block
class BlockNetwork(gluon.Block):
    def __init__(self, **kwargs):

        super(BlockNetwork , self).__init__(**kwargs)

        with self.name_scope():

            self.conv1 = gluon.nn.Conv2D(channels=60, kernel_size=(3, 3), strides=(1, 1), use_bias=True , activation="relu") # MNIST : result = ( batch size , 60 , 26 , 26) , CIFAR10 : : result = ( batch size , 60 , 30 , 30)
            self.maxpool1 = gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))  # MNIST : result = (batch size , 60 , 13 , 13) , CIFAR10 : result = (batch size , 60 , 15 , 15)
            self.conv2 = gluon.nn.Conv2D(channels=30, kernel_size=(6, 6), strides=(1, 1), use_bias=True , activation="relu")  # MNIST :  result = ( batch size , 30 , 8 , 8), CIFAR10 :  result = ( batch size , 30 , 10 , 10)
            self.maxpool2 = gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))  # MNIST : result = (batch size , 30 , 4 , 4) , CIFAR10 : result = (batch size , 30 , 5 , 5)
            self.fnn1 = gluon.nn.Dense(units=120 , activation="sigmoid" ,  use_bias=True, flatten=True)
            self.drop1 = gluon.nn.Dropout(0.2)
            self.fnn2 = gluon.nn.Dense(units=64 , activation="sigmoid",  use_bias=True)
            self.drop2 = gluon.nn.Dropout(0.2)
            self.fnn3 = gluon.nn.Dense(10, use_bias=True)

    def forward(self , x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.fnn1(x)
        x=self.drop1(x)
        x=self.fnn2(x)
        x=self.drop2(x)
        x=self.fnn3(x)
        return x

def CNN(epoch = 100 , batch_size=128, save_period=10 , load_period=100 ,optimizer="sgd",learning_rate= 0.01 , dataset = "MNIST", ctx=mx.gpu(0) , method=1):

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

    #Convolution Neural Network
    # formula : output_size=((input−weights+2*Padding)/Stride)+1
    # data size
    # MNIST,FashionMNIST = (batch size , 1 , 28 ,  28)
    # CIFAR = (batch size , 3 , 32 ,  32)

    '''note!!!
    To compile and optimize the HybridSequential, we can then call its hybridize method. 
    Only HybridBlocks, e.g. HybridSequential, can be compiled. But you can still call hybridize on normal Block 
    and its HybridBlock children will be compiled instead. We will talk more about HybridBlocks 
    '''

    if method==1:
        #method 1 : HybridBlock
        net=HybridBlockNetwork()
    elif method==2:
        #method 2 : Block
        net=BlockNetwork()
    else:
        #method 3 : using Sequential()
        net = gluon.nn.HybridSequential() # stacks 'Block's sequentially

        with net.name_scope():
            net.add(gluon.nn.Conv2D(channels=60 , kernel_size=(3,3) , strides=(1,1) , use_bias=True , activation="relu")) # MNIST : result = ( batch size , 60 , 26 , 26) , CIFAR10 : : result = ( batch size , 60 , 30 , 30)
            net.add(gluon.nn.MaxPool2D(pool_size=(2,2), strides=(2,2))) # MNIST : result = (batch size , 60 , 13 , 13) , CIFAR10 : result = (batch size , 60 , 15 , 15)
            net.add(gluon.nn.Conv2D(channels=30 , kernel_size=(6,6) , strides=(1,1) , use_bias=True , activation="relu")) # MNIST :  result = ( batch size , 30 , 8 , 8), CIFAR10 :  result = ( batch size , 30 , 10 , 10)
            net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))) # MNIST : result = (batch size , 30 , 4 , 4) , CIFAR10 : result = (batch size , 30 , 5 , 5)
            net.add(gluon.nn.Dense(units=120 , activation="sigmoid", use_bias=True , flatten=True))
            net.add(gluon.nn.Dropout(0.2))
            net.add(gluon.nn.Dense(units=64 , activation="sigmoid", use_bias=True))
            net.add(gluon.nn.Dropout(0.2))
            net.add(gluon.nn.Dense(10,use_bias=True))

    net.hybridize() # hybridize!!!! for faster learning - only for hybrid

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

            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)

            with autograd.record(train_mode=True):
                output=net(data)

                #loss definition
                '''Why do you write this?
                answer :  Blocks, sequential, softmaxCrossEntropyLoss, and other gluon package keywords should be accessed as classes by default.'''
                loss=gluon.loss.SoftmaxCrossEntropyLoss(axis=-1 , sparse_label=True )(output,label)
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

    test_accuracy = evaluate_accuracy(test_data , net , ctx)
    print("Test_acc : {}".format(test_accuracy))

    return "optimization completed"


if __name__ == "__main__":
    CNN(epoch = 100 , batch_size=128, save_period=10 , load_period=100 ,optimizer="sgd",learning_rate= 0.01 , dataset = "MNIST", ctx=mx.gpu(0) , method=1)
else :
    print("Imported")


