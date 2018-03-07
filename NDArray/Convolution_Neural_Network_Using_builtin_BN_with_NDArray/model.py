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
    test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(root="MNIST", train = False , transform = transform) ,128 , shuffle=False) #Loads data from a dataset and returns mini-batches of data.=128

    return train_data , test_data

#MFashionNIST dataset
def FashionMNIST(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label) # data normalization
    train_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = True , transform = transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.FashionMNIST(root="FashionMNIST" , train = False , transform = transform) ,128 , shuffle=False) #Loads data from a dataset and returns mini-batches of data.=128

    return train_data , test_data

#CIFAR10 dataset
def CIFAR10(batch_size):

    #transform = lambda data, label: (data.astype(np.float32) / 255.0 , label)
    train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(root="CIFAR10", train = True, transform=transform) , batch_size , shuffle=True , last_batch="rollover") #Loads data from a dataset and returns mini-batches of data.
    test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(root="CIFAR10", train = False, transform=transform) , 128 , shuffle=False) #Loads data from a dataset and returns mini-batches of data.=128

    return train_data , test_data

#evaluate the data
def evaluate_accuracy(data_iterator , network  , ctx):
    numerator = 0
    denominator = 0
    for data, label in data_iterator:

        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        output = network(data, drop_rate=0.0)  # is_training = False for using moving average , variance!!!

        predictions = nd.argmax(output, axis=1) # (batch_size , num_outputs)
        predictions=predictions.asnumpy()
        label=label.asnumpy()
        numerator += sum(predictions == label)
        denominator += data.shape[0]

    return (numerator / denominator)


def CNN(epoch = 100 , batch_size=256, save_period=10 , load_period=100 , weight_decay=0.001 ,learning_rate= 0.1 , dataset = "MNIST", ctx=mx.cpu(0)):

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
        [W1, B1, gamma1, beta1, W2, B2, gamma2, beta2, W3, B3, gamma3, beta3, W4, B4, gamma4, beta4, W5, B5, ma1, ma2, ma3, ma4, mv1, mv2, mv3, mv4]= nd.load(path)  # weights load

        ma1=ma1.as_in_context(ctx)
        ma2=ma2.as_in_context(ctx)
        ma3=ma3.as_in_context(ctx)
        ma4=ma4.as_in_context(ctx)
        mv1=mv1.as_in_context(ctx)
        mv2=mv2.as_in_context(ctx)
        mv3=mv3.as_in_context(ctx)
        mv4=mv4.as_in_context(ctx)
        W1=W1.as_in_context(ctx)
        B1=B1.as_in_context(ctx)
        gamma1=gamma1.as_in_context(ctx)
        beta1=beta1.as_in_context(ctx)
        W2=W2.as_in_context(ctx)
        B2=B2.as_in_context(ctx)
        gamma2=gamma2.as_in_context(ctx)
        beta2=beta2.as_in_context(ctx)
        W3=W3.as_in_context(ctx)
        B3=B3.as_in_context(ctx)
        gamma3=gamma3.as_in_context(ctx)
        beta3=beta3.as_in_context(ctx)
        W4=W4.as_in_context(ctx)
        B4=B4.as_in_context(ctx)
        gamma4=gamma4.as_in_context(ctx)
        beta4=beta4.as_in_context(ctx)
        W5=W5.as_in_context(ctx)
        B5=B5.as_in_context(ctx)

        MOVING_MEANS=[ma1,ma2,ma3,ma4]
        MOVING_VARS=[mv1,mv2,mv3,mv4]
        params = [W1 , B1 , gamma1 , beta1 , W2 , B2 , gamma2 , beta2 , W3 , B3 , gamma3 , beta3 , W4 , B4, gamma4 , beta4 , W5 , B5]

    else:

        print("initializing weights")
        weight_scale=0.01
        BN_weight_scale = 0.01

        with ctx:
            W1 = nd.random.normal(loc=0 , scale=weight_scale , shape=(60,color,3,3))
            B1 = nd.random.normal(loc=0 , scale=weight_scale , shape=60)

            gamma1 = nd.random.normal(shape=60, loc=1, scale=BN_weight_scale)
            beta1 = nd.random.normal(shape=60, scale=BN_weight_scale)

            ma1=nd.zeros(1)
            mv1=nd.zeros(1)

            W2 = nd.random.normal(loc=0 , scale=weight_scale , shape=(30,60,6,6))
            B2 = nd.random.normal(loc=0 , scale=weight_scale , shape=30)

            gamma2 = nd.random.normal(shape=30, loc=1, scale=BN_weight_scale)
            beta2 = nd.random.normal(shape=30, scale=BN_weight_scale)

            ma2=nd.zeros(1)
            mv2=nd.zeros(1)

            if dataset == "CIFAR10":
                reshape=750
            elif dataset == "MNIST" or dataset == "FashionMNIST":
                reshape=480

            W3 = nd.random.normal(loc=0 , scale=weight_scale , shape=(120, reshape))
            B3 = nd.random.normal(loc=0 , scale=weight_scale , shape=120)

            gamma3 = nd.random.normal(shape=120, loc=1, scale=BN_weight_scale)
            beta3 = nd.random.normal(shape=120, scale=BN_weight_scale)

            ma3=nd.zeros(1)
            mv3=nd.zeros(1)

            W4 = nd.random.normal(loc=0 , scale=weight_scale , shape=(64, 120))
            B4 = nd.random.normal(loc=0 , scale=weight_scale , shape=64)

            gamma4 = nd.random.normal(shape=64, loc=1, scale=BN_weight_scale)
            beta4 = nd.random.normal(shape=64, scale=BN_weight_scale)

            ma4=nd.zeros(1)
            mv4=nd.zeros(1)

            W5 = nd.random.normal(loc=0 , scale=weight_scale , shape=(num_outputs , 64))
            B5 = nd.random.normal(loc=0 , scale=weight_scale , shape=num_outputs)

        MOVING_MEANS=[ma1,ma2,ma3,ma4]
        MOVING_VARS=[mv1,mv2,mv3,mv4]
        params = [W1 , B1 , gamma1 , beta1 , W2 , B2 , gamma2 , beta2 , W3 , B3 , gamma3 , beta3 , W4 , B4, gamma4 , beta4 , W5 , B5]

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

    def network(X, drop_rate=0.0): # formula : output_size=((input−weights+2*Padding)/Stride)+1
        #data size
        # MNIST,FashionMNIST = (batch size , 1 , 28 ,  28)
        # CIFAR = (batch size , 3 , 32 ,  32)

        # builtin The BatchNorm function moving_mean, moving_var does not work.
        C_H1=nd.Activation(data=nd.BatchNorm(data=nd.Convolution(data=X , weight = W1 , bias = B1 , kernel=(3,3) , stride=(1,1) , num_filter=60), gamma=gamma1 , beta=beta1, moving_mean=ma1,moving_var=mv1,momentum=0.9,fix_gamma=False, use_global_stats=True) , act_type="relu") # MNIST : result = ( batch size , 60 , 26 , 26) , CIFAR10 : : result = ( batch size , 60 , 30 , 30)
        P_H1=nd.Pooling(data = C_H1 , pool_type = "avg" , kernel=(2,2), stride = (2,2)) # MNIST : result = (batch size , 60 , 13 , 13) , CIFAR10 : result = (batch size , 60 , 15 , 15)
        C_H2=nd.Activation(data=nd.BatchNorm(data=nd.Convolution(data=P_H1 , weight = W2 , bias = B2 , kernel=(6,6) , stride=(1,1) , num_filter=30), gamma=gamma2 , beta=beta2, moving_mean=ma2,moving_var=mv2,momentum=0.9,fix_gamma=False, use_global_stats=True), act_type="relu") # MNIST :  result = ( batch size , 30 , 8 , 8), CIFAR10 :  result = ( batch size , 30 , 10 , 10)
        P_H2=nd.Pooling(data = C_H2 , pool_type = "avg" , kernel=(2,2), stride = (2,2)) # MNIST : result = (batch size , 30 , 4 , 4) , CIFAR10 : result = (batch size , 30 , 5 , 5)
        P_H2 = nd.flatten(data=P_H2)

        '''FullyConnected parameter
        • data: (batch_size, input_dim)
        • weight: (num_hidden, input_dim)
        • bias: (num_hidden,)
        • out: (batch_size, num_hidden)
        '''
        F_H1 =nd.Activation(nd.BatchNorm(data=nd.FullyConnected(data=P_H2 , weight=W3 , bias=B3 , num_hidden=120), gamma=gamma3 , beta=beta3, moving_mean=ma3,moving_var=mv3,momentum=0.9,fix_gamma=False, use_global_stats=True),act_type="relu")
        F_H1 =nd.Dropout(data=F_H1, p=drop_rate)
        F_H2 =nd.Activation(nd.BatchNorm(data=nd.FullyConnected(data=F_H1 , weight=W4 , bias=B4 , num_hidden=64), gamma=gamma4 , beta=beta4, moving_mean=ma4,moving_var=mv4,momentum=0.9,fix_gamma=False, use_global_stats=True),act_type="relu")
        F_H2 =nd.Dropout(data=F_H2, p=drop_rate)
        #softmax_Y = nd.softmax(nd.FullyConnected(data=F_H2 ,weight=W5 , bias=B5 , num_hidden=10))
        out=nd.FullyConnected(data=F_H2 ,weight=W5 , bias=B5 , num_hidden=10)
        return out

    def softmax_cross_entropy(yhat_linear, y):
        return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)

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
                output = network(data, drop_rate=0.0)
                #loss definition
                #loss = cross_entropy(output,label) # (batch_size,)
                loss = softmax_cross_entropy(output,label) # (batch_size,)
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
                nd.save("weights/MNIST_weights-{}".format(i), params+MOVING_MEANS+MOVING_VARS)

            elif dataset=="CIFAR10":
                nd.save("weights/CIFAR10_weights-{}".format(i), params+MOVING_MEANS+MOVING_VARS)

            elif dataset=="FashionMNIST":
                nd.save("weights/FashionMNIST_weights-{}".format(i),params+MOVING_MEANS+MOVING_VARS)

    test_accuracy = evaluate_accuracy(test_data , network , ctx)
    print("Test_acc : {}".format(test_accuracy))

    return "optimization completed"

if __name__ == "__main__":
    CNN(epoch=100, batch_size=128, save_period=10 , load_period=100 , weight_decay=0.001 ,learning_rate=0.1, dataset="MNIST", ctx=mx.cpu(0))
else :
    print("Imported")


