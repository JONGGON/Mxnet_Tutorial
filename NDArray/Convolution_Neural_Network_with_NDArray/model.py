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
def evaluate_accuracy(data_iterator , network , ctx):
    numerator = 0
    denominator = 0

    for data, label in data_iterator:

        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = network(data)

        predictions = nd.argmax(output, axis=1) # (batch_size , num_outputs)

        predictions=predictions.asnumpy()
        label=label.asnumpy()
        numerator += sum(predictions == label)
        denominator += data.shape[0]

    return (numerator / denominator)

def CNN(epoch = 100 , batch_size=10, save_period=10 , load_period=100 , weight_decay=0.001 ,learning_rate= 0.1 , dataset = "MNIST", ctx=mx.cpu(0)):

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
    if dataset == "MNIST" or dataset =="FashionMNIST":
        color = 1
    elif dataset == "CIFAR10":
        color = 3
    num_outputs = 10

    if dataset == "MNIST":
        path = "weights/MNIST_weights-{}".format(load_period)
    elif dataset == "FashionMNIST":
        path = "weights/FashionMNIST_weights-{}".format(load_period)
    elif dataset == "CIFAR10":
        path = "weights/CIFAR10_weights-{}".format(load_period)

    if os.path.exists(path):
        print("loading weights")
        [W1, B1, W2, B2, W3, B3, W4, B4, W5, B5] = nd.load(path)  # weights load

        W1=W1.as_in_context(ctx)
        B1=B1.as_in_context(ctx)
        W2=W2.as_in_context(ctx)
        B2=B2.as_in_context(ctx)
        W3=W3.as_in_context(ctx)
        B3=B3.as_in_context(ctx)
        W4=W4.as_in_context(ctx)
        B4=B4.as_in_context(ctx)
        W5=W5.as_in_context(ctx)
        B5=B5.as_in_context(ctx)

        params = [W1 , B1 , W2 , B2 , W3 , B3 , W4 , B4 , W5 , B5]
    else:
        print("initializing weights")
        with ctx:
            W1 = nd.random.normal(loc=0 , scale=0.1 , shape=(60,color,3,3))
            B1 = nd.random.normal(loc=0 , scale=0.1 , shape=60)

            W2 = nd.random.normal(loc=0 , scale=0.1 , shape=(30,60,6,6))
            B2 = nd.random.normal(loc=0 , scale=0.1 , shape=30)

            if dataset == "CIFAR10":
                reshape=750
            elif dataset == "MNIST" or dataset == "FashionMNIST":
                reshape=480

            W3 = nd.random.normal(loc=0 , scale=0.1 , shape=(120, reshape))
            B3 = nd.random.normal(loc=0 , scale=0.1 , shape=120)

            W4 = nd.random.normal(loc=0 , scale=0.1 , shape=(64, 120))
            B4 = nd.random.normal(loc=0 , scale=0.1 , shape=64)

            W5 = nd.random.normal(loc=0 , scale=0.1 , shape=(num_outputs , 64))
            B5 = nd.random.normal(loc=0 , scale=0.1 , shape=num_outputs)

        params = [W1 , B1 , W2 , B2 , W3 , B3 , W4 , B4, W5 , B5]
        
    # attach gradient!!!
    for i, param in enumerate(params):
        param.attach_grad()

    # network - similar to lenet5 

    '''Convolution parameter
    data: (batch_size, channel, height, width)
    weight: (num_filter, channel, kernel[0], kernel[1])
    bias: (num_filter,)
    out: (batch_size, num_filter, out_height, out_width).
    '''

    def network(X,drop_rate=0.0): # formula : output_size=((input−weights+2*Padding)/Stride)+1
        #data size 
        # MNIST,FashionMNIST = (batch size , 1 , 28 ,  28)
        # CIFAR = (batch size , 3 , 32 ,  32)

        C_H1=nd.Activation(data= nd.Convolution(data=X , weight = W1 , bias = B1 , kernel=(3,3) , stride=(1,1)  , num_filter=60) , act_type="relu") # MNIST : result = ( batch size , 60 , 26 , 26) , CIFAR10 : : result = ( batch size , 60 , 30 , 30) 
        P_H1=nd.Pooling(data = C_H1 , pool_type = "max" , kernel=(2,2), stride = (2,2)) # MNIST : result = (batch size , 60 , 13 , 13) , CIFAR10 : result = (batch size , 60 , 15 , 15)
        C_H2=nd.Activation(data= nd.Convolution(data=P_H1 , weight = W2 , bias = B2 , kernel=(6,6) , stride=(1,1) , num_filter=30), act_type="relu") # MNIST :  result = ( batch size , 30 , 8 , 8), CIFAR10 :  result = ( batch size , 30 , 10 , 10)
        P_H2=nd.Pooling(data = C_H2 , pool_type = "max" , kernel=(2,2), stride = (2,2)) # MNIST : result = (batch size , 30 , 4 , 4) , CIFAR10 : result = (batch size , 30 , 5 , 5)
        P_H2 = nd.flatten(data=P_H2)

        '''FullyConnected parameter
        • data: (batch_size, input_dim)
        • weight: (num_hidden, input_dim)
        • bias: (num_hidden,)
        • out: (batch_size, num_hidden)
        '''
        F_H1 =nd.Activation(nd.FullyConnected(data=P_H2 , weight=W3 , bias=B3 , num_hidden=120),act_type="sigmoid")
        F_H1 =nd.Dropout(data=F_H1, p=drop_rate)
        F_H2 =nd.Activation(nd.FullyConnected(data=F_H1 , weight=W4 , bias=B4 , num_hidden=64),act_type="sigmoid")
        F_H2 =nd.Dropout(data=F_H2, p=drop_rate)
        softmax_Y = nd.softmax(nd.FullyConnected(data=F_H2 ,weight=W5 , bias=B5 , num_hidden=10))
        return softmax_Y

    def cross_entropy(output, label):
        return - nd.sum(label * nd.log(output), axis=1)

    #Adam optimizer
    state=[]
    optimizer=mx.optimizer.Adam(rescale_grad=1,learning_rate=learning_rate)
    for i,param in enumerate(params):
        state.append(optimizer.create_state(0,param))

    def SGD(params, lr , wd , bs):
        for param in params:
             param -= ((lr * param.grad)/bs+wd*param)

    for i in tqdm(range(1,epoch+1,1)):
        for data,label in train_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            label = nd.one_hot(label , num_outputs)

            with autograd.record():
                output = network(data,drop_rate=0.2)

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

    test_accuracy = evaluate_accuracy(test_data , network , ctx)
    print("Test_acc : {}".format(test_accuracy))

    return "optimization completed"

if __name__ == "__main__":
    CNN(epoch=100, batch_size=128, save_period=10 , load_period=100 , weight_decay=0.001 ,learning_rate=0.1, dataset="MNIST", ctx=mx.cpu(0))
else :
    print("Imported")


