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
def evaluate_accuracy(data_iterator , num_inputs , network , ctx , dataset):
    numerator = 0
    denominator = 0

    for data, label in data_iterator:

        if dataset=="CIFAR10":
            data = nd.slice_axis(data=data, axis=3, begin=0, end=1)

        data = data.as_in_context(ctx).reshape((-1,num_inputs))
        label = label.as_in_context(ctx)
        output = network(data,0.0) # when test , 'Dropout rate' must be 0.0

        predictions = nd.argmax(output, axis=1) # (batch_size , num_outputs)

        predictions=predictions.asnumpy()
        label=label.asnumpy()
        numerator += sum(predictions == label)
        denominator += data.shape[0]

    return (numerator / denominator)

def FNN(epoch = 100 , batch_size=10, save_period=10 , load_period=100 , weight_decay=0.001 ,learning_rate= 0.1 , dataset = "MNIST", ctx=mx.gpu(0)):

    #data selection
    if dataset =="MNIST":
        train_data , test_data = MNIST(batch_size)
    elif dataset == "CIFAR10":
        train_data, test_data = CIFAR10(batch_size)
    elif dataset == "FashionMNIST":
        train_data, test_data = FashionMNIST(batch_size)
    else:
        return "The dataset does not exist."

    # data structure
    if dataset == "MNIST" or dataset == "FashionMNIST":
        num_inputs = 28 * 28
    elif dataset == "CIFAR10":
        num_inputs = 32 * 32

    #network parameter    
    num_hidden = 50
    num_outputs = 10

    if dataset == "MNIST":
        path = "weights/MNIST_weights-{}".format(load_period)
    elif dataset == "FashionMNIST":
        path = "weights/FashionMNIST_weights-{}".format(load_period)
    elif dataset == "CIFAR10":
        path = "weights/CIFAR10_weights-{}".format(load_period)

    if os.path.exists(path):
        print("loading weights")
        [W1 , B1 , W2 , B2]= nd.load(path)  # weights load

        W1=W1.as_in_context(ctx)
        B1=B1.as_in_context(ctx)
        W2=W2.as_in_context(ctx)
        B2=B2.as_in_context(ctx)

        params = [W1 , B1 , W2 , B2]
    else:
        print("initializing weights")
        with ctx:
            W1 = nd.random.normal(loc=0, scale=0.01, shape=(num_inputs, num_hidden))
            B1 = nd.random.normal(loc=0, scale=0.01, shape=num_hidden)
            W2 = nd.random.normal(loc=0, scale=0.01, shape=(num_hidden, num_outputs))
            B2 = nd.random.normal(loc=0, scale=0.01, shape=num_outputs)

        params = [W1 , B1 , W2 , B2]

    # attach gradient!!!
    for i, param in enumerate(params):
        param.attach_grad()

    #Fully Neural Network with 1 Hidden layer 
    def network(X,dropout=0.0):
        H1 = nd.Activation(nd.dot(X, W1) + B1, act_type="relu")
        H1 = nd.Dropout(data=H1 , p=dropout) # apply dropout layer!!! 
        Y = nd.dot(H1, W2) + B2
        softmax_Y = nd.softmax(Y)
        return softmax_Y

    def cross_entropy(output, label):
        return - nd.sum(label * nd.log(output), axis=1)

    #Adam optimizer
    state=[]
    optimizer=mx.optimizer.Adam(rescale_grad=1,learning_rate=learning_rate)
    for i,param in enumerate(params):
        state.append(optimizer.create_state(0,param))

    #optimizer
    def SGD(params, lr , wd , bs):
        for param in params:
             param -= ((lr * param.grad)/bs+wd*param)

    for i in tqdm(range(1,epoch+1,1)):
        for data,label in train_data:
            if dataset == "CIFAR10":
                data = nd.slice_axis(data= data , axis=3 , begin = 0 , end=1)
            data = data.as_in_context(ctx).reshape((-1,num_inputs))
            label = label.as_in_context(ctx)
            label = nd.one_hot(label , num_outputs)

            with autograd.record():
                output = network(data,0.2)

                #loss definition
                loss = cross_entropy(output,label) # (batch_size,)
                cost = nd.mean(loss).asscalar()
            loss.backward()

            for j,param in enumerate(params):
                optimizer.update(0,param,param.grad,state[j])

            #SGD(params, learning_rate , weight_decay , batch_size)

        print(" epoch : {} , last batch cost : {}".format(i,cost))

        #weight_save
        if i % save_period==0:

            if not os.path.exists("weights"):
                os.makedirs("weights")

            print("saving weights")
            if dataset=="MNIST":
                nd.save("weights/MNIST_weights-{}".format(i),params)

            elif dataset=="CIFAR10":
                nd.save("weights/CIFAR10_weights-{}".format(i),params)

            elif dataset=="FashionMNIST":
                nd.save("weights/FashionMNIST_weights-{}".format(i),params)

    test_accuracy = evaluate_accuracy(test_data , num_inputs , network , ctx , dataset)
    print("Test_acc : {}".format(test_accuracy))

    return "optimization completed"

if __name__ == "__main__":
    FNN(epoch=100, batch_size=128, save_period=10 , load_period=100 , weight_decay=0.001 ,learning_rate=0.1, dataset="MNIST", ctx=mx.gpu(0))
else :
    print("Imported")


